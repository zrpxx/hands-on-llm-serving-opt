"""OpenAI 兼容的 HTTP 服务。

引擎运行在一个后台线程里，循环执行 step()；每个 HTTP 请求在事件循环里等待自己的 asyncio.Queue。
这与书中第 3 章的设计相同：API 层只负责收发，批处理与模型执行集中在一个循环里。

启动：
    python -m miniserve.server --model Qwen/Qwen3-0.6B 的本地路径 --port 8000
"""
import argparse
import asyncio
import json
import logging
import queue
import threading
import time
import uuid
from typing import AsyncIterator, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

from .config import EngineConfig
from .engine import LLMEngine, StepOutput
from .sequence import SamplingParams

logger = logging.getLogger("miniserve")


class AsyncEngine:
    """把同步的 LLMEngine 包装成可以在 asyncio 中使用的接口。"""

    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self.inbox: "queue.Queue" = queue.Queue()
        self.streams = {}                     # seq_id -> (asyncio.Queue, loop)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        # TODO(1-4)：
        # 后台线程主循环：没有任务时阻塞等待 inbox；处理完 inbox 中的全部命令后执行一次 engine.step()，
        # 把每个输出交给 _dispatch。
        # 分步指南第 10 步，测试：pytest tests/test_steps.py -k step10
        raise NotImplementedError("TODO(1-4): AsyncEngine._loop")

    def _handle(self, cmd):
        kind = cmd[0]
        if kind == "add":
            _, seq_id, prompt_ids, params, q, loop = cmd
            self.streams[seq_id] = (q, loop)
            try:
                self.engine.add_request(prompt_ids, params, seq_id)
            except ValueError as e:
                loop.call_soon_threadsafe(q.put_nowait, e)
                self.streams.pop(seq_id, None)
        elif kind == "abort":
            self.engine.abort_request(cmd[1])
            self.streams.pop(cmd[1], None)

    def _dispatch(self, out: StepOutput):
        # TODO(1-4)：
        # 把一个 StepOutput 投递到对应请求的 asyncio.Queue（注意跨线程要用 loop.call_soon_threadsafe）；
        # 请求结束后从 self.streams 中移除。
        # 分步指南第 10 步，测试：pytest tests/test_steps.py -k step10
        raise NotImplementedError("TODO(1-4): AsyncEngine._dispatch")

    async def generate(self, prompt_ids: List[int], params: SamplingParams, seq_id: str) -> AsyncIterator:
        q: asyncio.Queue = asyncio.Queue()
        self.inbox.put(("add", seq_id, prompt_ids, params, q, asyncio.get_running_loop()))
        try:
            while True:
                item = await q.get()
                if isinstance(item, Exception):
                    raise item
                yield item
                if item[1]:
                    return
        finally:
            self.inbox.put(("abort", seq_id))      # 客户端断开时释放 KV 块；已完成的请求重复 abort 无副作用


class IncrementalDetokenizer:
    """逐 token 输出文本。多字节字符可能跨 token，解码出替换字符时先不输出。"""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.ids: List[int] = []
        self.text = ""

    def push(self, token_id: int) -> str:
        self.ids.append(token_id)
        text = self.tok.decode(self.ids, skip_special_tokens=True)
        if text.endswith("�"):
            return ""
        delta, self.text = text[len(self.text):], text
        return delta


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: Union[str, List[int]]
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    ignore_eos: bool = False
    stream_options: Optional[dict] = None


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[dict]
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    ignore_eos: bool = False
    stream_options: Optional[dict] = None


def create_app(engine_cfg: EngineConfig, model_name: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="MiniServe")
    engine = LLMEngine(engine_cfg)
    aengine = AsyncEngine(engine)
    tokenizer = AutoTokenizer.from_pretrained(engine_cfg.model)
    served_name = model_name or engine_cfg.model

    def params_of(r) -> SamplingParams:
        return SamplingParams(max_tokens=r.max_tokens, temperature=r.temperature, top_p=r.top_p,
                              ignore_eos=r.ignore_eos)

    async def run(prompt_ids: List[int], r, chat: bool):
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:16]
        created = int(time.time())
        obj = "chat.completion" if chat else "text_completion"
        detok = IncrementalDetokenizer(tokenizer)
        gen = aengine.generate(prompt_ids, params_of(r), rid)
        usage = lambda n: {"prompt_tokens": len(prompt_ids), "completion_tokens": n,
                           "total_tokens": len(prompt_ids) + n}

        def chunk(delta: str, finish: Optional[str]):
            choice = {"index": 0, "finish_reason": finish}
            if chat:
                choice["delta"] = {"content": delta} if delta else {}
            else:
                choice["text"] = delta
            return {"id": rid, "object": obj + (".chunk" if chat else ""), "created": created,
                    "model": served_name, "choices": [choice]}

        if r.stream:
            async def sse():
                n = 0
                if chat:
                    first = chunk("", None)
                    first["choices"][0]["delta"] = {"role": "assistant"}
                    yield f"data: {json.dumps(first)}\n\n"
                async for tok, finished, reason in gen:
                    n += 1
                    yield f"data: {json.dumps(chunk(detok.push(tok), reason if finished else None), ensure_ascii=False)}\n\n"
                if r.stream_options and r.stream_options.get("include_usage"):
                    yield f"data: {json.dumps({'id': rid, 'object': obj, 'choices': [], 'usage': usage(n)})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        n, reason = 0, None
        try:
            async for tok, finished, reason in gen:
                n += 1
                detok.push(tok)
        except ValueError as e:
            raise HTTPException(400, str(e))
        choice = {"index": 0, "finish_reason": reason}
        if chat:
            choice["message"] = {"role": "assistant", "content": detok.text}
        else:
            choice["text"] = detok.text
        return JSONResponse({"id": rid, "object": obj, "created": created, "model": served_name,
                             "choices": [choice], "usage": usage(n)})

    @app.post("/v1/completions")
    async def completions(r: CompletionRequest):
        ids = r.prompt if isinstance(r.prompt, list) else tokenizer.encode(r.prompt, add_special_tokens=False)
        return await run(ids, r, chat=False)

    @app.post("/v1/chat/completions")
    async def chat(r: ChatRequest):
        out = tokenizer.apply_chat_template(r.messages, add_generation_prompt=True, tokenize=True)
        ids = out["input_ids"] if not isinstance(out, list) else out
        return await run(list(ids), r, chat=True)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": served_name, "object": "model", "owned_by": "miniserve"}]}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        s, sch = engine.stats, engine.scheduler
        return {"running": len(sch.running), "waiting": len(sch.waiting),
                "free_blocks": engine.allocator.num_free, "total_blocks": engine.num_blocks,
                "preemptions": sch.total_preemptions, **s}

    app.state.engine = engine
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--served-model-name", default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--num-gpu-blocks", type=int, default=0)
    ap.add_argument("--policy", choices=["continuous", "static"], default="continuous")
    ap.add_argument("--no-chunked-prefill", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = EngineConfig(model=a.model, device=a.device, dtype=a.dtype, block_size=a.block_size,
                       max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_num_batched_tokens,
                       max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_memory_utilization,
                       num_gpu_blocks=a.num_gpu_blocks, policy=a.policy, chunked_prefill=not a.no_chunked_prefill)
    import uvicorn
    uvicorn.run(create_app(cfg, a.served_model_name), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
