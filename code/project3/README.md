# 项目 3　Kernel Forge

说明文档见站点中的「项目 3 Kernel Forge」页面（`stage3/boss-kernel-forge.md`），每一步的提示都在那里。

```bash
pip install -r requirements.txt
FORGE_IMPL=solution python -m pytest -q tests     # 参考实现，应全部通过
python -m pytest -q tests -k step01               # 你的实现，逐步完成

# 投机解码实验（需要项目 2 训练的小模型 /tmp/tiny-lm 和语料 ../project2/data）
python ../project2/tools/train_tiny_lm.py --data ../project2/data --out /tmp/draft-lm \
    --tokenizer /tmp/tiny-lm --layers 1 --hidden 128
python spec_demo.py --target /tmp/tiny-lm --draft /tmp/draft-lm --data ../project2/data

python bench.py                                   # 需要 NVIDIA GPU
```

没有 GPU 时，测试会自动设置 `TRITON_INTERPRET=1`，在 CPU 上解释执行 kernel。
