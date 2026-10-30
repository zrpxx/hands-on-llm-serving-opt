"""项目 4 Cluster Commander：一个推理集群的离散事件模拟器。

模拟的对象是"若干个推理实例 + 一个路由器"：每个实例按连续批处理、chunked prefill 的方式运行，
有自己的前缀缓存；每一步的耗时由 Roofline 估算（costmodel.py）。用它可以在 CPU 上几秒钟内比较
不同的路由策略、PD 分离的配置，以及它们对 TTFT、ITL 和缓存命中率的影响。
"""
