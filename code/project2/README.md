# 项目 2　Quant Arena

说明文档见站点中的「项目 2 Quant Arena」页面（`stage2/boss-quant-arena.md`），六个步骤的手算例子和提示都在那里。

```bash
pip install -r requirements.txt
QUANTLAB_IMPL=solution python -m pytest -q tests     # 参考实现，应全部通过
python -m pytest -q tests -k step01                  # 你的实现，逐步完成

python tools/make_corpus.py --out data
python tools/train_tiny_lm.py --data data --out /tmp/tiny-lm
python run.py --model /tmp/tiny-lm --data data --method gptq --bits 3
```

`run.py` 和 `tools/` 下的脚本默认使用 `starter/` 中你的实现；设置 `QUANTLAB_IMPL=solution` 使用参考实现（`prune_24.py` 总是使用参考实现的评测函数）。
