"""把正文中的「X.Y 节」「项目 N」变成指向对应页面的链接。可以重复运行，已经是链接的不会再处理。

    python scripts/link_sections.py            # 修改文件
    python scripts/link_sections.py --dry-run  # 只统计
"""
import argparse
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
SIDEBAR = (ROOT / "_sidebar.md").read_text(encoding="utf-8")

SECTIONS = {}
for title, path in re.findall(r"\[([^\]]+)\]\(([^)]+\.md)\)", SIDEBAR):
    m = re.match(r"(\d\.\d{1,2}) ", title)
    if m:
        SECTIONS[m.group(1)] = path
    m = re.match(r"项目 (\d)\s+\S", title)
    if m and "walkthrough" not in path:
        SECTIONS["项目" + m.group(1)] = path

NUMS = r"\d\.\d{1,2}"
REF = re.compile(rf"(?<![\d.\[])((?:{NUMS}\s*[、和与及]\s*)*{NUMS})(\s*节)")
PROJ = re.compile(r"(?<![\[\w])项目 ([1-4])(?![\d\]])")


def link_nums(group: str, self_path: str) -> str:
    def one(m):
        n = m.group(0)
        p = SECTIONS.get(n)
        if not p or p == self_path:
            return n
        return f"[{n}]({p})"
    return re.sub(NUMS, one, group)


def process_line(line: str, self_path: str, state: dict) -> str:
    # 跳过标题、测验组件、HTML 行
    if line.lstrip().startswith(("#", "<div", "<details", "<summary", "|-", "| -")) or "data-questions" in line:
        return line
    # 按行内代码与已有链接切分，只处理普通文本片段
    parts = re.split(r"(`[^`]*`|\[[^\]]*\]\([^)]*\)|\$[^$]*\$)", line)
    out = []
    for i, seg in enumerate(parts):
        if i % 2 == 1:
            out.append(seg)
            continue
        def ref(m):
            nums, tail = m.group(1), m.group(2)
            p = SECTIONS.get(nums)
            if p and p != self_path:            # 单个节号：把「节」也放进链接
                return f"[{nums}{tail}]({p})"
            return link_nums(nums, self_path) + tail
        seg = REF.sub(ref, seg)

        def proj(m):
            key = "项目" + m.group(1)
            p = SECTIONS.get(key)
            if not p or p == self_path or key in state["proj"]:
                return m.group(0)
            state["proj"].add(key)
            return f"[项目 {m.group(1)}]({p})"
        seg = PROJ.sub(proj, seg)
        out.append(seg)
    return "".join(out)


def process(path: pathlib.Path) -> tuple[str, int]:
    rel = path.relative_to(ROOT).as_posix()
    text = path.read_text(encoding="utf-8")
    lines, out, fence = text.split("\n"), [], False
    state = {"proj": {"项目" + n for n in re.findall(r"\[项目 ([1-4])\]\(", text)}}
    for line in lines:
        if line.lstrip().startswith("```"):
            fence = not fence
            out.append(line)
            continue
        out.append(line if fence else process_line(line, rel, state))
    new = "\n".join(out)
    return new, new.count("](") - text.count("](")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    files = sorted(list(ROOT.glob("stage*/*.md")) + list(ROOT.glob("final/*.md")) + list(ROOT.glob("guide/*.md"))
                   + list(ROOT.glob("appendix/*.md")))
    total = 0
    for f in files:
        new, n = process(f)
        if n:
            total += n
            print(f"{n:4d}  {f.relative_to(ROOT)}")
            if not a.dry_run:
                f.write_text(new, encoding="utf-8")
    print(f"共新增 {total} 个链接")


if __name__ == "__main__":
    main()
