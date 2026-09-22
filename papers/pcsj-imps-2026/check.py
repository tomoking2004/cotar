"""PCSJ/IMPS 予稿の書式検査——組版して，そのまま提出できる形かを機械的に確かめる．

    python check.py          組版し，全項目を検査して結果を出す（manuscript.pdf を更新する）
    python check.py --hook   Claude Code の PostToolUse フックから呼ぶ．tex と sty が前回合格時から
                             変わっていなければ何もしない．不合格なら報告を stderr に出して終了コード 2

検査項目の出どころは三つ——公式サンプル（template/）の定め，アップロード画面の定め，
../principles.md §3・§6 の記法と走査．合格すると .check-ok に tex と sty のハッシュを書く（git には載せない）．
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "manuscript.tex"
STY = HERE / "pcsjimps-j.sty"
DVI = HERE / "manuscript.dvi"
PDF = HERE / "manuscript.pdf"
TEMPLATE_STY = HERE / "template" / "samplefile2015" / "pcsjimps-j.sty"
STAMP = HERE / ".check-ok"
TOOLS = ("platex", "dvipdfmx", "pdfinfo", "pdffonts")

# 公式サンプルの定め
MAX_PAGES = 2                       # 1件2ページ以内（図表を含む）
A4 = "595.28 x 841.89"              # A4（pdfinfo の表記）
ABSTRACT_CHARS = 300                # 和文アブストラクトは 300 字程度（超過は警告のみ）
# アップロード画面の定め
MAX_PDF_BYTES = 3 * 1024 * 1024     # 3MB を超えると更新されない
REGISTERED_TITLE = "タスク類似性に基づく内部表現整合学習が視覚言語モデルの出力に与える影響"
REGISTERED_AUTHORS = ("中野 友晴", "杉村 大輔")
# principles.md の定め
LABEL_PREFIXES = ("chap", "sec", "tab", "fig", "eq")
SPACING_EXCEPTION = r"〒\hspace{0pt}"   # 公式サンプル由来の定型だけは手動空白を許す


def run(*args: str) -> subprocess.CompletedProcess[str]:
    """原稿のディレクトリで外部コマンドを走らせ，出力を文字列で受ける．"""
    return subprocess.run(args, cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")


def fingerprint() -> str:
    """tex と sty の内容のハッシュ．合格の印（.check-ok）に書く．"""
    h = hashlib.sha256()
    for p in (TEX, STY):
        h.update(p.read_bytes())
    return h.hexdigest()


def typeset() -> list[str]:
    """platex 2回 → dvipdfmx -p a4 で PDF を作る．エラー・警告・バッドボックスは 0 でなければならない．"""
    fails: list[str] = []
    for _ in range(2):
        if run("platex", "-interaction=nonstopmode", TEX.name).returncode != 0:
            break
    log = (HERE / "manuscript.log").read_text(encoding="utf-8", errors="replace").splitlines()
    for label, prefix in (("TeX エラー", "! "), ("バッドボックス", ("Overfull", "Underfull")), ("LaTeX Warning", "LaTeX Warning")):
        if hits := [l for l in log if l.startswith(prefix)]:
            fails.append(f"{label} {len(hits)} 件: " + " / ".join(hits[:3]))
    if DVI.exists():
        r = run("dvipdfmx", "-p", "a4", DVI.name)
        if r.returncode != 0:
            fails.append("dvipdfmx が失敗した: " + r.stderr.strip()[-200:])
    else:
        fails.append("DVI が書き出されていない")
    return fails


def inspect_pdf() -> list[str]:
    """提出物の PDF を見る——ページ数・用紙・暗号化・サイズ・フォント埋め込み．"""
    if not PDF.exists():
        return ["manuscript.pdf が無い"]
    fails: list[str] = []
    info = run("pdfinfo", PDF.name).stdout
    pages = int(m.group(1)) if (m := re.search(r"Pages:\s+(\d+)", info)) else 0
    if pages > MAX_PAGES:
        fails.append(f"{pages} ページある（上限 {MAX_PAGES} ページ）")
    if A4 not in info:
        fails.append("用紙が A4 でない")
    if not re.search(r"Encrypted:\s+no", info):
        fails.append("暗号化・パスワード・編集制限がかかっている")
    if (size := PDF.stat().st_size) > MAX_PDF_BYTES:
        fails.append(f"{size / 2**20:.1f}MB ある（上限 3MB）")
    fonts = [f.split() for f in run("pdffonts", PDF.name).stdout.splitlines()[2:]]
    if not_embedded := [f[0] for f in fonts if f and f[-5] != "yes"]:
        fails.append("埋め込まれていないフォント: " + ", ".join(not_embedded))
    return fails


def inspect_source() -> tuple[list[str], list[str]]:
    """原稿の tex を見る——文字と記法・登録内容との一致・文献の並び・アブストラクトの長さ．"""
    fails: list[str] = []
    warns: list[str] = []
    src = TEX.read_text(encoding="utf-8")
    lines = src.splitlines()

    def where(pattern: str) -> list[int]:
        return [i for i, l in enumerate(lines, 1) if re.search(pattern, l.replace(SPACING_EXCEPTION, ""))]

    for message, pattern in (
        ("禁止文字（、。 全角英数 半角カナ 〜）", r"[、。]|[０-９Ａ-Ｚａ-ｚ]|[ｦ-ﾟ]|〜"),
        ("手動空白 \\vspace/\\hspace（〒\\hspace{0pt} 以外）", r"\\[vh]space"),
        (f"ラベル接頭辞が規約外（{'/'.join(LABEL_PREFIXES)}:）", r"\\label\{(?!(" + "|".join(LABEL_PREFIXES) + r"):)"),
        ("図表の配置に [h] がある（[t]・[b] を使う）", r"\\begin\{(table|figure)\}\[[^\]]*[hH]"),
        ("ⓒ・copyright の記載がある", r"(?i)\(c\)|©|ⓒ|copyright"),
    ):
        if hits := where(pattern):
            fails.append(f"{message}: 行 {hits}")

    title = re.sub(r"\\\\|\s", "", m.group(1)) if (m := re.search(r"\\JTitle\{(.*?)\}\s*\n", src, re.S)) else ""
    if title != REGISTERED_TITLE:
        fails.append(f"和文タイトルが登録内容と違う: 「{title}」")
    authors = tuple(re.sub(r"\$.*$", "", a).strip() for a in re.findall(r"\\JEAuthor\{([^}]*)\}", src))
    if authors != REGISTERED_AUTHORS:
        fails.append(f"著者名が登録内容と違う: {authors}")

    cites = list(dict.fromkeys(k.strip() for keys in re.findall(r"\\cite\{([^}]*)\}", src) for k in keys.split(",")))
    bibs = re.findall(r"\\bibitem\{([^}]*)\}", src)
    if cites != bibs:
        fails.append(f"文献の並びが本文初出順と違う: 初出 {cites} / 文献 {bibs}")

    if m := re.search(r"\\Abstract\{\s*(.*?)\s*\}\s*\]", src, re.S):
        if (n := len(re.sub(r"\s", "", m.group(1)))) > ABSTRACT_CHARS + 30:
            warns.append(f"アブストラクトが {n} 字（{ABSTRACT_CHARS} 字程度）")
    else:
        fails.append("\\Abstract{...} が見つからない")
    return fails, warns


def check() -> tuple[list[str], list[str]]:
    """全項目を検査し，(不合格の理由, 警告) を返す．不合格が空なら提出できる形である．"""
    fails: list[str] = []
    if STY.read_bytes() != TEMPLATE_STY.read_bytes():
        fails.append("pcsjimps-j.sty が公式テンプレート（template/samplefile2015/）と異なる")
    fails += typeset()
    fails += inspect_pdf()
    source_fails, warns = inspect_source()
    fails += source_fails
    for ext in ("aux", "log", "dvi", "out"):
        (HERE / f"manuscript.{ext}").unlink(missing_ok=True)
    return fails, warns


def report(fails: list[str], warns: list[str]) -> str:
    """人が読む結果．不合格は x，警告は ! で並べる．"""
    head = (f"予稿の書式検査: 不合格 {len(fails)} 件（{TEX.relative_to(HERE.parent.parent).as_posix()}）" if fails
            else f"予稿の書式検査: 合格（{MAX_PAGES} ページ以内・A4・全フォント埋め込み・警告 0）")
    return "\n".join([head, *(f"  x {f}" for f in fails), *(f"  ! {w}" for w in warns)])


def main() -> int:
    """終了コード：合格 0，不合格 1（フックからは 2——Claude Code が報告を差し戻す）．"""
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    hook = "--hook" in sys.argv
    if hook:
        try:
            json.load(sys.stdin)          # フックの入力は読み捨てる（変更の有無は指紋で見る）
        except ValueError:
            pass
        if STAMP.exists() and STAMP.read_text().strip() == fingerprint():
            return 0
    if missing := [t for t in TOOLS if shutil.which(t) is None]:
        print("PATH に無い: " + ", ".join(missing), file=sys.stderr)
        return 2 if hook else 1
    fails, warns = check()
    if fails:
        STAMP.unlink(missing_ok=True)
        print(report(fails, warns), file=sys.stderr)
        return 2 if hook else 1
    STAMP.write_text(fingerprint())
    print(report(fails, warns))
    return 0


if __name__ == "__main__":
    sys.exit(main())
