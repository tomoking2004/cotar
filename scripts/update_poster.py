"""Write the robustness measurement into the poster. No GPU, no torch.

Section 3.3 of the poster reports `probe_drift.py`, and context.md §8.1 reads that result only
once seeds 45 and 46 are in. Which sentence the poster may say was fixed before those seeds
ran, as a table of what each outcome means, so choosing it is not a judgement left for the
day: this reads `analyses/probe-drift.json`, finds the row of §8.1 the numbers fall in, and
writes that row's wording and the values the two charts of 3.3 show. Run it after pulling
the results.

It touches only what the measurement decides — the two charts, the ratio and the share of
answers that change, and the paragraphs `slots` names — and at the level of the parts, as
presentations/principles.md §5 asks: the slide, the charts and their embedded workbooks are
edited in place, and nothing else in the package is rewritten. Everything else on the poster,
including anything edited by hand in PowerPoint, stays as it is. Section 3.2 keeps to the
three seeds of the main results (§8.1).

A paragraph is left alone when its text already reads as it should, so running this twice
changes nothing, and "nothing to change" doubles as the check that the poster agrees with
`analyses/`. The research document comes first: this writes what §8.1 says the row means,
so the poster agrees with the document only while §8.1 and §A.6 record the same row.

Needs `lxml` (the `poster` extra) and nothing from the package, so `pip install lxml` is
enough on a machine without torch, where the poster is opened.
"""

from __future__ import annotations

import copy
import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, NamedTuple

from lxml import etree

ROOT        = Path(__file__).resolve().parent.parent
DRIFT_PATH  = ROOT / "analyses" / "probe-drift.json"
POSTER_PATH = ROOT / "presentations" / "20261116_PCSJ-IMPS2026_ポスター.pptx"

# The decision quantity of §A.6: the ratio to the image-swap control, over every variant,
# at the layer the alignment is applied to.
FAMILY  = "any"
LAYER   = "16"
CONTROL = "other_image"
ARMS    = ("baseline", "proposal", "ablation")

# The poster is one hue: conclusions stand out by size, not by colour.
GREEN     = "4A8C45"
ACCENT    = GREEN
BOLD_FONT = "游ゴシック"
MATH_FONT = "Cambria Math"

SLIDE      = "ppt/slides/slide1.xml"
SLIDE_RELS = "ppt/slides/_rels/slide1.xml.rels"
NS         = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
}
MARKUP = re.compile(r"\{([gr]):(.*?)\}")
SYMBOL = re.compile("(𝒉)")


@dataclass(frozen=True)
class Drift:
    """What the poster takes from the measurement."""

    seeds: int
    ratio: dict[str, float]          # arm → ratio to the control, %
    distance: dict[str, float]       # arm → distance under perturbation
    control: dict[str, float]        # arm → distance under the control
    ratio_gap: dict[str, Any]        # proposal − baseline
    ablation_gap: dict[str, Any]     # ablation − baseline
    distance_gap: dict[str, Any]     # proposal − baseline
    flips: dict[str, float]          # arm → share of (question, perturbation) pairs that flip
    answers_differ: bool             # does any pair of arms differ in how often answers flip?
    answers_close: bool              # do all their intervals stay within ±2pt?


class Reading(NamedTuple):
    """One row of §8.1, in the poster's words."""

    row: str
    meaning: str
    conclusion: str


def read_drift(path: Path = DRIFT_PATH) -> Drift:
    """The numbers of 3.3, read from `probe_drift.py`'s report."""
    report = json.loads(path.read_text(encoding="utf-8"))
    family = report["comparison"][FAMILY]
    ratio, distance = family[LAYER][f"{CONTROL}/ratio_mean"], family[LAYER]["drift_mean"]
    control = {
        arm: fmean(run["controls"][CONTROL][LAYER]["mean"] for run in report["runs"][arm].values())
        for arm in ARMS
    }
    return Drift(
        seeds=len(report["seeds"]),
        ratio={arm: ratio["arms"][arm]["mean"] for arm in ARMS},
        distance={arm: distance["arms"][arm]["mean"] for arm in ARMS},
        control=control,
        ratio_gap=ratio["differences"]["proposal - baseline"],
        ablation_gap=ratio["differences"]["ablation - baseline"],
        distance_gap=distance["differences"]["proposal - baseline"],
        flips={arm: family["flip_rate"]["arms"][arm]["mean"] for arm in ARMS},
        answers_differ=any(d["excludes_zero"] for d in family["flip_rate"]["differences"].values()),
        answers_close=all(max(-d["ci_low"], d["ci_high"]) * 100 <= 2
                          for d in family["flip_rate"]["differences"].values()),
    )


def reading_of(drift: Drift) -> Reading:
    """The row of §8.1 the proposal's ratio falls in, split by what the distance does."""
    ratio, distance, ablation = drift.ratio_gap, drift.distance_gap, drift.ablation_gap
    if not ratio["excludes_zero"]:
        return Reading(
            "区間が 0 を含む",
            "動きの比の差ははっきりしない",
            "𝒉 が加工で動きにくくなったとは確認できない",
        )

    # The row asks that the ablation not shrink too; an ablation whose interval holds zero
    # has not been shown to, and the row stands.
    falls = ratio["mean"] < 0
    sign = "負" if falls else "正"
    moved = "小さくなった" if falls else "大きくなった"
    if falls and ablation["excludes_zero"] and ablation["mean"] < 0:
        return Reading(
            "区間が 0 を含まず，負．ただし ablation でも小さくなる",
            "ablation でも動きの比が低下",
            "動きの比の低下は，署名の表す手順には帰せない",
        )
    if not distance["excludes_zero"]:
        return Reading(
            f"区間が 0 を含まず，{sign}．距離の向きは決められない",
            "距離そのものの差ははっきりしない",
            f"𝒉 の動きの比は{moved}",
        )
    if (distance["mean"] < 0) != falls:
        return Reading(
            f"区間が 0 を含まず，{sign}．距離は逆向き",
            f"𝒉 が動き{'にくく' if falls else 'やすく'}なったとまでは言えない",
            f"𝒉 の動きの比は{moved}が，",
        )
    if falls:
        return Reading(
            "区間が 0 を含まず，負．距離も同じ向き",
            "動きの比でも距離でも小",
            "署名の表す手順で整合したことにより，𝒉 が加工で動きにくくなった",
        )
    return Reading(
        "区間が 0 を含まず，正．距離も同じ向き",
        "動きの比でも距離でも大",
        "違う手順を遠ざけた副作用により，𝒉 が加工で動きやすくなった",
    )


def times(value: float) -> str:
    """A ratio in the poster's words: whole times from two up, one decimal below."""
    return f"約{round(value)}倍" if value >= 2 else f"約{value:.1f}倍"


def ratio_line(drift: Drift) -> str:
    """What the ratio tells, not where proposal ranks.

    It names the ratio, beside the discussion that says how far 𝒉 moves at all.
    """
    gap = drift.ratio_gap
    if not gap["excludes_zero"]:
        return "動きの比の差ははっきりしない"
    change = "縮小" if gap["mean"] < 0 else "拡大"
    return f"比で見た 𝒉 の動きが{change}"


def ratio_arms(drift: Drift) -> str:
    """Which arms moved: proposal alone, or the ablation with it."""
    gap, ablation = drift.ratio_gap, drift.ablation_gap
    if not gap["excludes_zero"]:
        return "proposal と baseline の差は区間が 0 を含む"
    word = "低下" if gap["mean"] < 0 else "上昇"
    both = ablation["excludes_zero"] and (ablation["mean"] < 0) == (gap["mean"] < 0)
    return f"proposal と ablation で{word}" if both else f"proposal だけ{word}"


def moved_more(drift: Drift) -> str:
    """How far 𝒉 moves at all in proposal, against baseline: under perturbation and the control.

    The ratio divides the one by the other, so the two together say why it fell or rose.
    """
    perturbed = drift.distance["proposal"] / drift.distance["baseline"]
    swapped = drift.control["proposal"] / drift.control["baseline"]
    larger, smaller = min(perturbed, swapped) > 1, max(perturbed, swapped) < 1
    relation = "より大きい" if larger else "より小さい" if smaller else "と違う"
    times_each = f"加工で{times(perturbed)}，別の画像で{times(swapped)}"
    return f"proposal は 𝒉 の動き自体が baseline {relation}．{times_each}"


def slots(drift: Drift) -> dict[tuple[str, int], str]:
    """Each paragraph the measurement decides, by shape name and position in the shape.

    How far each arm moved is the charts' to show, so no paragraph gives the differences
    (presentations/principles.md §2).
    """
    reading = reading_of(drift)
    if drift.answers_differ:
        answers = "群のあいだに差"
    else:
        answers = "3群でほとんど同じ" if drift.answers_close else "差ははっきりしない"
    return {
        ("Drift result", 0): f"{{g:結果}}\t{ratio_line(drift)}",
        ("Drift result", 1): ratio_arms(drift),
        ("Drift answers", 0): f"{{g:結果}}\t{answers}",
        ("Drift discussion", 0): f"{{g:考察}}\t{moved_more(drift)}",
        ("Drift discussion", 1): f"{{r:{reading.conclusion}}}",
        ("Drift discussion", 2): f"{{r:{reading.meaning}}}",
    }


def plain(markup: str) -> str:
    """The text of a slot's markup, with its colour marks taken out."""
    return MARKUP.sub(r"\2", markup)


def paragraph_text(paragraph: etree._Element) -> str:
    """What a paragraph reads, across all its runs."""
    return "".join(t.text or "" for t in paragraph.findall(".//a:t", NS))


def restyle(style: etree._Element, color: str | None, font: str | None) -> etree._Element:
    """A copy of a run's properties in another colour (and so the real bold) or typeface."""
    style = copy.deepcopy(style)
    if color:
        style.set("b", "1")
        font = font or BOLD_FONT
        fill = style.find("a:solidFill", NS)
        if fill is None:
            fill = etree.Element(f"{{{NS['a']}}}solidFill")
            style.insert(0, fill)
        for old in list(fill):
            fill.remove(old)
        etree.SubElement(fill, f"{{{NS['a']}}}srgbClr").set("val", color)
    if font:
        for tag in ("latin", "ea", "cs"):
            face = style.find(f"a:{tag}", NS)
            if face is None:
                face = etree.SubElement(style, f"{{{NS['a']}}}{tag}")
            face.set("typeface", font)
    return style


def set_paragraph(paragraph: etree._Element, markup: str) -> None:
    """Replace the runs, keeping the look of the paragraph's last run in the body font."""
    runs = paragraph.findall("a:r", NS)
    body = [r for r in runs if not SYMBOL.fullmatch(r.findtext("a:t", "", NS))]
    style = body[-1].find("a:rPr", NS)
    colors = {"g": GREEN, "r": ACCENT}

    pieces, position = [], 0
    for match in MARKUP.finditer(markup):
        pieces.append((markup[position:match.start()], None))
        pieces.append((match[2], colors[match[1]]))
        position = match.end()
    pieces.append((markup[position:], None))

    index = paragraph.index(runs[0])
    for run in runs:
        paragraph.remove(run)
    for content, color in pieces:
        for part in filter(None, SYMBOL.split(content)):
            run = etree.Element(f"{{{NS['a']}}}r")
            font = MATH_FONT if SYMBOL.fullmatch(part) else None
            run.append(restyle(style, color, font))
            etree.SubElement(run, f"{{{NS['a']}}}t").text = part
            paragraph.insert(index, run)
            index += 1


def shape_named(slide: etree._Element, name: str) -> etree._Element:
    """The shape PowerPoint's selection pane lists under `name`."""
    for found in slide.iter(f"{{{NS['p']}}}cNvPr"):
        if found.get("name") == name:
            return found.getparent().getparent()
    raise KeyError(f"no shape named {name!r} on the poster")


def write_slots(slide: etree._Element, drift: Drift) -> list[str]:
    """Rewrite each paragraph of `slots` that does not already read as it should."""
    changes = []
    for (name, index), markup in slots(drift).items():
        paragraph = shape_named(slide, name).findall("p:txBody/a:p", NS)[index]
        was = paragraph_text(paragraph)
        if was != plain(markup):
            set_paragraph(paragraph, markup)
            changes.append(f"{name}: {was} → {plain(markup)}")
    return changes


def chart_parts(parts: dict[str, bytes], slide: etree._Element, name: str) -> tuple[str, str]:
    """The chart part behind a graphic frame, and the workbook embedded in it."""
    frame = shape_named(slide, name)
    rid = frame.find(".//c:chart", NS).get(f"{{{NS['r']}}}id")
    rels = etree.fromstring(parts[SLIDE_RELS])
    target = next(r.get("Target") for r in rels if r.get("Id") == rid)
    chart = "ppt/" + target.removeprefix("../")
    stem = chart.rsplit("/", 1)[1]
    rels = etree.fromstring(parts[f"ppt/charts/_rels/{stem}.rels"])
    workbook = next(r.get("Target") for r in rels if r.get("Type").endswith("/package"))
    return chart, "ppt/" + workbook.removeprefix("../")


def write_workbook(data: bytes, values: list[float]) -> bytes:
    """Put the values in column B of the embedded sheet, under its header row."""
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(out, "w") as target:
        for item in source.infolist():
            content = source.read(item.filename)
            if item.filename == "xl/worksheets/sheet1.xml":
                sheet = etree.fromstring(content)
                for row, value in enumerate(values, start=2):
                    cell = sheet.find(f".//x:c[@r='B{row}']/x:v", NS)
                    cell.text = repr(value)
                content = etree.tostring(sheet, xml_declaration=True, encoding="UTF-8",
                                         standalone=True)
            target.writestr(item, content, zipfile.ZIP_DEFLATED)
    return out.getvalue()


def shown(values: list[float]) -> str:
    """A chart's values as the change report prints them."""
    return "／".join(f"{value:.2f}" for value in values)


def write_chart(parts: dict[str, bytes], slide: etree._Element, name: str,
                values: dict[str, float], top: int) -> list[str]:
    """Put the arms' values and the axis top into a chart and its workbook, if either differs."""
    chart_name, workbook = chart_parts(parts, slide, name)
    chart = etree.fromstring(parts[chart_name])
    points = chart.findall(".//c:val/c:numRef/c:numCache/c:pt/c:v", NS)
    scale = chart.find(".//c:valAx/c:scaling/c:max", NS)
    was = [float(p.text) for p in points]
    now = [values[arm] for arm in ARMS]
    same = all(math.isclose(a, b, rel_tol=1e-9) for a, b in zip(was, now, strict=True))
    if same and scale.get("val") == str(top):
        return []

    for point, value in zip(points, now, strict=True):
        point.text = repr(value)
    was_top = scale.get("val")
    scale.set("val", str(top))
    parts[chart_name] = etree.tostring(chart, xml_declaration=True, encoding="UTF-8",
                                       standalone=True)
    parts[workbook] = write_workbook(parts[workbook], now)
    return [f"{name}: {shown(was)} → {shown(now)}, scale up to {was_top} → {top}"]


def percent_scale(parts: dict[str, bytes], slide: etree._Element) -> int:
    """The top of the value axis of 3.2's charts in %, which 3.3's charts take too."""
    chart_name, _ = chart_parts(parts, slide, "Chart accuracy")
    top = etree.fromstring(parts[chart_name]).find(".//c:valAx/c:scaling/c:max", NS)
    return int(top.get("val"))


def update(poster: Path = POSTER_PATH, drift_path: Path = DRIFT_PATH) -> list[str]:
    """Bring the poster in line with the measurement, and say what moved."""
    drift = read_drift(drift_path)
    with zipfile.ZipFile(poster) as package:
        infos = package.infolist()
        parts = {info.filename: package.read(info.filename) for info in infos}

    slide = etree.fromstring(parts[SLIDE])
    changes = write_slots(slide, drift)
    # Every chart in % shares one scale across 3.2 and 3.3, whose bars stand on one base line
    # (principles.md §4): the scale 3.2's charts already carry.
    answers = {arm: 100 * share for arm, share in drift.flips.items()}
    top = percent_scale(parts, slide)
    changes += write_chart(parts, slide, "Chart ratio", drift.ratio, top)
    changes += write_chart(parts, slide, "Chart answers", answers, top)
    if changes:
        parts[SLIDE] = etree.tostring(slide, xml_declaration=True, encoding="UTF-8",
                                      standalone=True)
        with zipfile.ZipFile(poster, "w") as package:
            for info in infos:
                package.writestr(info, parts[info.filename], zipfile.ZIP_DEFLATED)

    print(f"{drift.seeds} seeds. §8.1: {reading_of(drift).row}")
    print("\n".join(changes) if changes else "nothing to change: the poster agrees with analyses/")
    return changes


if __name__ == "__main__":
    update()
