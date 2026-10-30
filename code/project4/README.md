# 项目 4　Cluster Commander

说明文档见站点中的「项目 4 Cluster Commander」页面（`stage4/boss-cluster-commander.md`）。

一个推理集群的离散事件模拟器，只依赖 Python 标准库和 pytest。

```bash
pip install -r requirements.txt
CLUSTER_IMPL=solution python -m pytest -q tests   # 参考实现，应全部通过
python -m pytest -q tests -k step01              # 你的实现，逐步完成

python run.py routers      # 比较路由策略
python run.py pd           # 聚合部署与 PD 分离
python run.py ratio        # 不同的 P:D 配比
```
