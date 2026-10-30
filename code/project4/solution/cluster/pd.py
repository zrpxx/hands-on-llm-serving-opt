"""第 4 步：PD 分离。对应 4.3 节。

请求先到 prefill 实例，算完 prompt、产出第一个 token 后，把 KV 传给一个 decode 实例，由它生成后续的 token。
"""
from .router import LeastLoad
from .sim import Cluster


def kv_transfer_time(input_len: int, model, link_gbs: float, layerwise: bool = False, latency: float = 2e-4) -> float:
    """把一个请求的 KV 从 prefill 实例传到 decode 实例需要多久（秒）。
    layerwise=True 时，每算完一层就传这一层的 KV，传输与后面各层的计算重叠，
    prefill 结束后只剩最后一层的数据还要传。"""
    t = input_len * model.kv_bytes_per_token / (link_gbs * 1e9)
    if layerwise:
        t /= model.layers
    return t + latency


class DisaggCluster(Cluster):
    def __init__(self, prefill_instances, decode_instances, router, layerwise: bool = True):
        super().__init__(prefill_instances, router)
        self.decode = decode_instances
        self.decode_router = LeastLoad()
        self.layerwise = layerwise
        self.transfer_times = []          # 每个请求从 prefill 结束到 KV 全部到达 decode 实例的时间
        self.link_free = {}               # 每个 prefill 实例的出口链路什么时候空闲

    def all_instances(self):
        return self.instances + self.decode

    def on_prefill_done(self, req, now):
        """prefill 实例算完了 req：选一个 decode 实例，安排 KV 传输。
        每个 prefill 实例的出口链路同一时间只能传一个请求的 KV，后来的要排队。
        逐层传输时，前 L-1 层的 KV 已经在 prefill 计算期间传完，结束后只剩最后一层；
        但如果链路还被前一个请求占着，传输也只能等它结束后才开始。"""
        src = next(i for i in self.instances if i.iid == req.instance)
        full = kv_transfer_time(req.input_len, src.model, src.hw.link_gbs, layerwise=False, latency=0)
        free = self.link_free.get(src.iid, 0.0)
        if self.layerwise:
            end = max(now + full / src.model.layers, free + full)
        else:
            end = max(now, free) + full
        end += 2e-4                                       # 建立连接、通知等固定延迟
        self.link_free[src.iid] = end
        self.transfer_times.append(end - now)
        dst = self.decode[self.decode_router.choose(req, self.decode)]
        self.push(end, "kv_arrived", (req, dst))

    def on_event(self, kind, payload, now):
        if kind != "kv_arrived":
            raise ValueError(kind)
        req, dst = payload
        dst.enqueue_decode(req, now)
        self._kick(dst, now)
