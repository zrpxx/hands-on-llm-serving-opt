import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, os.environ.get("CLUSTER_IMPL", "starter")))


class FakeInstance:
    """只提供路由器需要的负载信号，用来单独测试路由策略。"""

    def __init__(self, queued=0, hit=0, n=0):
        self._q, self._hit, self._n = queued, hit, n

    def queued_tokens(self):
        return self._q

    def num_requests(self):
        return self._n

    def cached_prefix_tokens(self, req):
        return self._hit
