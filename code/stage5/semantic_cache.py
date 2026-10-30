"""一个最小的语义缓存，用来观察"相似"与"答案相同"之间的差距。对应 5.3 节。

真实系统用嵌入模型（如 BGE、ModernBERT 系列）把问题变成向量；这里为了不依赖任何模型，用汉字二元组的
TF-IDF 向量代替，相似度同样用余弦。它能识别字面相近的问题，却分不清"只差一个地名"的问题，
也认不出换了说法的同一个问题。嵌入模型在后一点上好得多，但前一类错误依然存在。

    python code/stage5/semantic_cache.py --threshold 0.6
"""
import argparse
import math
from collections import Counter

# 缓存中已有的问题和答案
CACHE = [
    ("北京飞上海要多久", "约 2 小时 15 分钟"),
    ("怎么重置登录密码", "在登录页点击忘记密码，按邮件中的链接重置"),
    ("Python 怎么读取 JSON 文件", "用 json.load(open(path))"),
    ("今天北京天气怎么样", "（实时数据，不应缓存）"),
    ("退货的运费谁出", "质量问题由商家承担，其他情况由买家承担"),
]
# 新请求，以及它的正确答案应该对应缓存中的哪一条（None 表示缓存里没有正确答案）
QUERIES = [
    ("从北京飞到上海需要多长时间", 0),
    ("北京飞广州要多久", None),          # 只差一个城市，答案不同
    ("北京飞上海要多久？经济舱", 0),
    ("我忘记密码了，怎么办", 1),
    ("怎么重置支付密码", None),          # 字面很像，业务完全不同
    ("Python 怎么写入 JSON 文件", None),
    ("python 读 json", 2),
    ("明天北京天气怎么样", None),        # 时间不同，而且本来就不该缓存
    ("退货运费由谁承担", 4),
    ("换货的运费谁出", None),
]


def bigrams(s):
    s = "".join(ch.lower() for ch in s if not ch.isspace() and ch not in "，。？！,.?!")
    return Counter(s[i:i + 2] for i in range(len(s) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.6)
    a = ap.parse_args()
    docs = [bigrams(q) for q, _ in CACHE] + [bigrams(q) for q, _ in QUERIES]
    df = Counter(g for d in docs for g in d)
    idf = {g: math.log((1 + len(docs)) / (1 + c)) + 1 for g, c in df.items()}

    def vec(s):
        v = {g: c * idf.get(g, 1.0) for g, c in bigrams(s).items()}
        n = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {g: x / n for g, x in v.items()}

    cache_vecs = [vec(q) for q, _ in CACHE]
    right = wrong = missed = 0
    print(f"阈值 {a.threshold}\n")
    for q, truth in QUERIES:
        v = vec(q)
        sims = [sum(v.get(g, 0) * x for g, x in c.items()) for c in cache_vecs]
        best = max(range(len(CACHE)), key=lambda i: sims[i])
        hit = sims[best] >= a.threshold
        if hit and best == truth:
            verdict, right = "命中，答案正确", right + 1
        elif hit:
            verdict, wrong = f"命中，但答案错误（返回了：{CACHE[best][1]}）", wrong + 1
        elif truth is not None:
            verdict, missed = "未命中，本可以复用", missed + 1
        else:
            verdict = "未命中，正确"
        print(f"{sims[best]:.2f}  {q:<22} → 最像「{CACHE[best][0]}」  {verdict}")
    print(f"\n正确命中 {right}，错误命中 {wrong}，错过 {missed}")


if __name__ == "__main__":
    main()
