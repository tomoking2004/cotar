# このリポジトリで作業するときの決まり

## 学会予稿（papers/pcsj-imps-2026/）は，いつでも提出できる形に保つ

- 書式の定めは公式テンプレート [papers/pcsj-imps-2026/template/](papers/pcsj-imps-2026/template/) と [papers/principles.md](papers/principles.md) にある．手元の `pcsjimps-j.sty` は公式と同一でなければならず，書き換えない．
- `manuscript.tex` や `pcsjimps-j.sty` を変えたら，必ず [papers/pcsj-imps-2026/check.py](papers/pcsj-imps-2026/check.py) を通す．PostToolUse フック（[.claude/settings.json](.claude/settings.json)）が Edit・Write・Bash の後に自動で走り，不合格なら理由を返す．**不合格のまま作業を終えない．** 合格した組版が `manuscript.pdf` であり，これが提出物である．
- ページ上限に収める手段の順序は principles.md §1 のとおり——本文の推敲が先で，見出しの間隔や行送りの書き換えには進まない．
- アブストラクトは教授が了承した文である．変えるときは本人の判断を仰ぐ．

## 外部への連絡・提出・登録は，絶対に勝手に行わない

- 学会事務局・教授・申込フォーム・原稿アップロード用 URL（トークン付きの ken.ieice.org/ken/form/…）への送信やアクセスをしない．メールに書かれた個別 URL・連絡先は本人だけが扱う．
- 返信文・登録内容は「案」として本文に示す．送るのは本人である．
