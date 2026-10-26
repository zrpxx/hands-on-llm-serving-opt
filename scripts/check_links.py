"""检查 Docsify 站点里所有指向本地 .md 的链接是否存在。

Docsify 配置了 relativePath: false，所以链接都按站点根目录解析。
用法：python scripts/check_links.py   （在仓库根目录运行）
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", ".github", "_site", "_to_delete", "node_modules", "scripts"}
LINK = re.compile(r"\]\(([^)\s]+?\.md)(?:[?#][^)]*)?\)")

missing = []
pages = [
    p for p in ROOT.rglob("*.md")
    if not SKIP_DIRS.intersection(p.relative_to(ROOT).parts)
]
for page in pages:
    text = page.read_text(encoding="utf-8")
    text = re.sub(r"```.*?```", "", text, flags=re.S)  # 忽略代码块里的示例
    for target in LINK.findall(text):
        if target.startswith(("http://", "https://")):
            continue
        if not (ROOT / target.lstrip("/")).is_file():
            missing.append(f"{page.relative_to(ROOT)} -> {target}")

if missing:
    print(f"发现 {len(missing)} 个失效链接：")
    print("\n".join(f"  {m}" for m in missing))
    sys.exit(1)
print(f"OK：检查了 {len(pages)} 个页面，没有失效链接。")
