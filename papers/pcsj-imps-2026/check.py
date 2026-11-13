"""PCSJ/IMPS 予稿の書式検査——組版して，そのまま提出できる形かを機械的に確かめる．

    python check.py          組版し，全項目を検査して結果を出す（manuscript.pdf を更新する）
    python check.py --hook   Claude Code の PostToolUse フックから呼ぶ．組版はしない．
                             manuscript.tex・pcsjimps-j.sty・manuscript.pdf が提出版の
                             コミットと同一なら何も出さずに終了コード 0，違えば，どれが
                             違うかと戻し方を stderr に出して終了コード 2

検査項目の出どころは三つ——公式サンプル（template/）の定め，アップロード画面の定め，
../principles.md §3・§6 の記法と走査．

フックが組版しないのは，締切を過ぎた原稿を変えないためである．組み直せば，アップロードした
PDF と同じ tex からでも manuscript.pdf が書き換わりうる．フックは提出版との一致だけを見て，
誤って触ったときに知らせる．
"""

from __future__ import annotations

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
TOOLS = ("platex", "dvipdfmx", "pdfinfo", "pdffonts")
# 提出版：教授の了承を得て 2026-10-16 にアップロードしたコミット．フックはこれと比べる
SUBMITTED = "70fb342"
SUBMITTED_FILES = (TEX, STY, PDF)

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
    return subprocess.run(
        args, cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False,  # 失敗はここで投げない．呼び手が returncode を見る
    )


def drifted_from_submission() -> list[Path] | None:
    """提出版（SUBMITTED）と中身が違うファイル．比べられないときは None．

    git の blob の名で比べる（作業ツリーの git hash-object と git rev-parse SUBMITTED:<path>）．
    git が載せる形に直してから名を付けるので，作業ツリーの改行が CRLF でも同じ名になる．
    """
    if shutil.which("git") is None:
        return None
    # 提出版のコミットが無い（浅い clone など）と，ここが失敗する
    wanted = run("git", "rev-parse", *(f"{SUBMITTED}:./{p.name}" for p in SUBMITTED_FILES))
    if wanted.returncode != 0:
        return None
    present = [p for p in SUBMITTED_FILES if p.exists()]
    hashed = run("git", "hash-object", "--", *(p.name for p in present))
    if hashed.returncode != 0:
        return None
    have = dict(zip(present, hashed.stdout.split(), strict=True))
    return [
        p for p, want in zip(SUBMITTED_FILES, wanted.stdout.split(), strict=True)
        if have.get(p) != want
    ]


def typeset() -> list[str]:
    """platex 2回 → dvipdfmx -p a4 で PDF を作る．

    エラー・警告・バッドボックスは 0 でなければならない．
    """
    fails: list[str] = []
    for _ in range(2):
        if run("platex", "-interaction=nonstopmode", TEX.name).returncode != 0:
            break
    log = (HERE / "manuscript.log").read_text(encoding="utf-8", errors="replace").splitlines()
    for label, prefix in (
        ("TeX エラー", "! "),
        ("バッドボックス", ("Overfull", "Underfull")),
        ("LaTeX Warning", "LaTeX Warning"),
    ):
        if hits := [line for line in log if line.startswith(prefix)]:
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
        return [
            i for i, line in enumerate(lines, 1)
            if re.search(pattern, line.replace(SPACING_EXCEPTION, ""))
        ]

    for message, pattern in (
        ("禁止文字（、。 全角英数 半角カナ 〜）", r"[、。]|[０-９Ａ-Ｚａ-ｚ]|[ｦ-ﾟ]|〜"),
        ("手動空白 \\vspace/\\hspace（〒\\hspace{0pt} 以外）", r"\\[vh]space"),
        (f"ラベル接頭辞が規約外（{'/'.join(LABEL_PREFIXES)}:）",
         r"\\label\{(?!(" + "|".join(LABEL_PREFIXES) + r"):)"),
        ("図表の配置に [h] がある（[t]・[b] を使う）", r"\\begin\{(table|figure)\}\[[^\]]*[hH]"),
        ("ⓒ・copyright の記載がある", r"(?i)\(c\)|©|ⓒ|copyright"),
    ):
        if hits := where(pattern):
            fails.append(f"{message}: 行 {hits}")

    title = (
        re.sub(r"\\\\|\s", "", m.group(1))
        if (m := re.search(r"\\JTitle\{(.*?)\}\s*\n", src, re.DOTALL))
        else ""
    )
    if title != REGISTERED_TITLE:
        fails.append(f"和文タイトルが登録内容と違う: 「{title}」")
    authors = tuple(
        re.sub(r"\$.*$", "", a).strip() for a in re.findall(r"\\JEAuthor\{([^}]*)\}", src)
    )
    if authors != REGISTERED_AUTHORS:
        fails.append(f"著者名が登録内容と違う: {authors}")

    cites = list(dict.fromkeys(
        k.strip() for keys in re.findall(r"\\cite\{([^}]*)\}", src) for k in keys.split(",")
    ))
    bibs = re.findall(r"\\bibitem\{([^}]*)\}", src)
    if cites != bibs:
        fails.append(f"文献の並びが本文初出順と違う: 初出 {cites} / 文献 {bibs}")

    if m := re.search(r"\\Abstract\{\s*(.*?)\s*\}\s*\]", src, re.DOTALL):
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
    path = TEX.relative_to(HERE.parent.parent).as_posix()
    head = (f"予稿の書式検査: 不合格 {len(fails)} 件（{path}）" if fails
            else f"予稿の書式検査: 合格（{MAX_PAGES} ページ以内・A4・全フォント埋め込み・警告 0）")
    return "\n".join([head, *(f"  x {f}" for f in fails), *(f"  ! {w}" for w in warns)])


def hook() -> int:
    """フックの入口．提出版と違うファイルがあれば，理由を stderr に出して 2 を返す．

    PostToolUse で終了コード 2 を返すと，Claude Code は stderr を Claude に差し戻す（道具は
    既に走った後なので，止めるのでなく知らせる）．比べられないとき（git が無い，浅い clone で
    提出版のコミットが無い）は黙って 0 を返す——フックは道具を使うたびに走るので，判定できない
    ことを毎回言えば本物の知らせが埋もれる．提出版を守るのは CLAUDE.md の決まりで，フックは
    その見張りにすぎない．
    """
    try:
        json.load(sys.stdin)              # フックの入力は読み捨てる（変わったかは中身で見る）
    except ValueError:
        pass
    if not (drifted := drifted_from_submission()):
        return 0
    root = HERE.parent.parent
    paths = [p.relative_to(root).as_posix() for p in drifted]
    print("\n".join([
        f"予稿が提出版（{SUBMITTED}）と違う: {', '.join(paths)}",
        "締切を過ぎた原稿は変えない（CLAUDE.md）．提出版に戻すには:",
        *(f"  git checkout {SUBMITTED} -- {path}" for path in paths),
    ]), file=sys.stderr)
    return 2


def main() -> int:
    """終了コード：合格 0，不合格 1．--hook は hook() の定めに従う．"""
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    if "--hook" in sys.argv:
        return hook()
    if missing := [t for t in TOOLS if shutil.which(t) is None]:
        print("PATH に無い: " + ", ".join(missing), file=sys.stderr)
        return 1
    fails, warns = check()
    if fails:
        print(report(fails, warns), file=sys.stderr)
        return 1
    print(report(fails, warns))
    return 0


if __name__ == "__main__":
    sys.exit(main())
