"""第 4 步：PD 分离。对应 4.3 节。

请求先到 prefill 实例，算完 prompt、产出第一个 token 后，把 KV 传给一个 decode 实例，由它生成后续的 token。
"""
from .router import LeastLoad
from .sim import Cluster


def kv_transfer_time(input_len: int, model, link_gbs: float, layerwise: bool = False, latency: float = 2e-4) -> float:
    """把一个请求的 KV 从 prefill 实例传到 decode 实例需要多久（秒）。
    layerwise=True 时，每算完一层就传这一层的 KV，传输与后面各层的计算重叠，
    prefill 结束后只剩最后一层的数据还要传。"""
    # TODO(第 4 步)：KV 的字节数 = input_len × 每个 token 的 KV 字节数；除以带宽（GB/s 换算成字节/秒）。
    # 逐层传输时，只剩 1/层数 的数据暴露在关键路径上。最后加上 latency。
    raise NotImplementedError("TODO(第 4 步): kv_transfer_time")


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
        # TODO(第 4 步)：
        # 1. full = 整体传输这个请求的 KV 要多久（kv_transfer_time，layerwise=False，latency=0）；
        # 2. 这条链路 self.link_free.get(src.iid, 0.0) 之前不空闲。逐层传输时，结束时间为
        #    max(now + full / 层数, 链路空闲时间 + full)；整体传输时为 max(now, 链路空闲时间) + full；再加 2e-4；
        # 3. 更新 self.link_free[src.iid]，把 结束时间 - now 记入 self.transfer_times；
        # 4. 用 self.decode_router 在 self.decode 中选一个实例，self.push(结束时间, "kv_arrived", (req, 实例))。
        raise NotImplementedError("TODO(第 4 步): on_prefill_done")

    def on_event(self, kind, payload, now):
        if kind != "kv_arrived":
            raise ValueError(kind)
        req, dst = payload
        dst.enqueue_decode(req, now)
        self._kick(dst, now)
