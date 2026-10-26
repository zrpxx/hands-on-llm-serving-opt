"""基线服务：用 Hugging Face Transformers 逐个处理请求（无批处理）。

对应项目 1 的步骤 1-1。接口与 MiniServe 相同，可以直接用 bench.py 压测。
    python baseline_hf.py --model PATH --port 8001               # 使用 KV Cache
    python baseline_hf.py --model PATH --port 8001 --no-kv-cache  # 每步重算全部上下文
"""
import argparse
import asyncio
import json
import threading
import time
import uuid
from typing import List, Optional, Union

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: Union[str, List[int]]
    max_tokens: int = 128
    temperature: float = 0.0
    stream: bool = False
    ignore_eos: bool = False
    stream_options: Optional[dict] = None


def create_app(model_path: str, device: str, use_cache: bool) -> FastAPI:
    tok = AutoTokenizer.from_pretrained(model_path)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device).eval()
    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id])
    lock = threading.Lock()          # 一次只处理一个请求
    app = FastAPI()

    @torch.inference_mode()
    def generate(ids: List[int], r: CompletionRequest, emit):
        with lock:
            seq = torch.tensor([ids], device=device)
            past, inp = None, seq
            for i in range(r.max_tokens):
                out = model(input_ids=inp, past_key_values=past, use_cache=use_cache)
                logits = out.logits[0, -1].float()
                if r.temperature > 0:
                    nxt = torch.multinomial(torch.softmax(logits / r.temperature, -1), 1).item()
                else:
                    nxt = int(logits.argmax())
                done = (not r.ignore_eos and nxt in eos) or i == r.max_tokens - 1
                emit(nxt, done)
                if done:
                    return
                if use_cache:
                    past, inp = out.past_key_values, torch.tensor([[nxt]], device=device)
                else:
                    seq = torch.cat([seq, torch.tensor([[nxt]], device=device)], dim=1)
                    inp = seq

    @app.post("/v1/completions")
    async def completions(r: CompletionRequest):
        ids = r.prompt if isinstance(r.prompt, list) else tok.encode(r.prompt, add_special_tokens=False)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        threading.Thread(target=generate, args=(ids, r, lambda t, d: loop.call_soon_threadsafe(q.put_nowait, (t, d))),
                         daemon=True).start()
        rid, created = "cmpl-" + uuid.uuid4().hex[:16], int(time.time())

        async def tokens():
            while True:
                t, d = await q.get()
                yield t, d
                if d:
                    return

        if r.stream:
            async def sse():
                n = 0
                async for t, d in tokens():
                    n += 1
                    c = {"id": rid, "object": "text_completion", "created": created,
                         "choices": [{"index": 0, "text": tok.decode([t]), "finish_reason": "length" if d else None}]}
                    yield f"data: {json.dumps(c)}\n\n"
                if r.stream_options and r.stream_options.get("include_usage"):
                    u = {"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n}
                    yield f"data: {json.dumps({'id': rid, 'choices': [], 'usage': u})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")
        out = [t async for t, _ in tokens()]
        return JSONResponse({"id": rid, "object": "text_completion", "created": created,
                             "choices": [{"index": 0, "text": tok.decode(out), "finish_reason": "length"}],
                             "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out)}})

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--no-kv-cache", action="store_true")
    a = ap.parse_args()
    import uvicorn
    uvicorn.run(create_app(a.model, a.device, not a.no_kv_cache), host="0.0.0.0", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
