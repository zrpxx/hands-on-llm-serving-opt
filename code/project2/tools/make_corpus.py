"""生成训练小模型和计算困惑度用的语料，不需要联网。

两部分来源：本教程第 0、1 部分的中文正文（去掉代码块、HTML 和公式），以及本机 Python 标准库源码里的
英文文档字符串。后者只用来扩充数据量，让小模型不至于过拟合。按段落 9:1 切分训练集和测试集。
    python tools/make_corpus.py --out data
"""
import argparse
import ast
import sysconfig
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def clean(md: str) -> str:
    md = re.sub(r"```.*?```", "", md, flags=re.S)
    md = re.sub(r"<[^>]+>", "", md)
    md = re.sub(r"\$\$.*?\$\$", "", md, flags=re.S)
    md = re.sub(r"\$[^$\n]+\$", "", md)
    md = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md)
    md = re.sub(r"[#>*|`]", "", md)
    lines = [ln.strip() for ln in md.splitlines()]
    return "\n".join(ln for ln in lines if len(ln) > 8 and not set(ln) <= set("-: "))


def python_docs(limit_chars: int):
    root = sysconfig.get_paths()["stdlib"]
    seen, out, total = set(), [], 0
    for f in sorted(glob.glob(os.path.join(root, "*.py")) + glob.glob(os.path.join(root, "*", "*.py"))):
        if "/test" in f or "idlelib" in f or "site-packages" in f:
            continue
        try:
            tree = ast.parse(open(f, encoding="utf-8").read())
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                d = ast.get_docstring(node)
                if d and len(d) > 150 and d not in seen:
                    seen.add(d)
                    out.append(" ".join(d.split()))
                    total += len(d)
                    if total > limit_chars:
                        return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--python-docs-mb", type=float, default=2.0, help="从标准库文档字符串中取多少 MB 文本")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(ROOT, "stage0", "*.md")) + glob.glob(os.path.join(ROOT, "stage1", "*.md")))
    paras = []
    for f in files:
        paras += clean(open(f, encoding="utf-8").read()).split("\n")
    paras += python_docs(int(a.python_docs_mb * 1e6))
    test = paras[::10]
    train = [p for i, p in enumerate(paras) if i % 10]
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "train.txt"), "w").write("\n".join(train))
    open(os.path.join(a.out, "test.txt"), "w").write("\n".join(test))
    print(f"{len(paras)} 段，训练集 {sum(map(len, train))} 字符，测试集 {sum(map(len, test))} 字符")


if __name__ == "__main__":
    main()
