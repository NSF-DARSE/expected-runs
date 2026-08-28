"""Render the coach-facing comparison page from the emitted JSON. No number is typed here.

Reads coach_page_data.json (thirds tables, spreads, paired bootstraps) and
coach_model_disagreement.json (the Location+ blind-spot exhibit) from the workdir and
writes a single self-contained HTML file. Minimal prose by design: the page is spoken to,
so it carries tables, charts, and labels rather than narrative.

BRANDING (UD Athletics standards): UD Blue #00539f primary, Light Blue #00a0df for
component scores, Cool Gray #bdbdbd for context/baselines, UD Yellow #ffd200 as the
single highlight accent. Oswald for display type, Open Sans for body, both with system
fallbacks so the page renders offline. White background, no gridlines, direct labels,
no dual axes, no rainbow ramp.

SIGN CONVENTION: fastball criterion values are run value RELATIVE TO AVERAGE per 100
fastballs, LOWER = BETTER, and every table on the page states that. Spread is
worst-third minus best-third, so BIGGER = sorts better.

Data rules: reads workdir JSON only; writes the HTML to the workdir (outside the repo).
Contains no pitcher names -- group means and counts only. Never committed.
"""
from __future__ import annotations

import json
import math
import os

import pandas as pd

import coach_model_paired as cp
import fair_criterion as fc

BLUE, LIGHT, GRAY, YELLOW, NAVY = "#00539f", "#00a0df", "#bdbdbd", "#ffd200", "#003c71"
INK, MUTED = "#1a1a1a", "#5c5c5c"
GOOD_HEX = "#1a7f4f"   # same green as the heatmap's GOOD, needed before the CSS f-string
REF = "#8a8a8a"  # the "last year's results" reference: neutral, but darker than a baseline
COLOR = {"velo_hi": GRAY, "coach_hi": GRAY, "stuff_hi": LIGHT, "loc_hi": LIGHT,
         "prior_hi": REF, "pitch2_hi": BLUE}
# Coach-facing labels, overriding the analysis labels carried in the JSON. Presentation
# lives here so a rename never requires rebuilding the data.
# CAUTION on "v1 Stuff+": that is the hand-weighted scorecard. It collides with the
# repo's own history, where CLAUDE.md's v1 is the swing-only whiff/SHAP model, a
# different thing. "v2 Stuff+" does match the repo's v2 (the fitted ridge). These names
# are deliberately confined to this file and never propagated into an analysis script.
LABELS = {"velo_hi": "Velocity only", "coach_hi": "v1 Stuff+", "stuff_hi": "Stuff+",
          "loc_hi": "Location+", "prior_hi": "Last year's run value",
          "pitch2_hi": "Stuff+ and Location+"}
BANDS = ["best third", "middle", "worst third"]
BAND_NOTE = {"best third": "graded best third", "middle": "graded middle third",
             "worst third": "graded worst third"}

CSS = f"""
*{{box-sizing:border-box}}
body{{margin:0;padding:44px 40px 60px;background:#fff;color:{INK};
 font-family:'Open Sans','Segoe UI',Helvetica,Arial,sans-serif;font-size:15px;line-height:1.5}}
.wrap{{max-width:1120px;margin:0 auto}}
h1{{font-family:Oswald,'Arial Narrow',Impact,sans-serif;font-weight:600;font-size:31px;
 letter-spacing:.01em;margin:0 0 6px;color:{NAVY};text-transform:uppercase}}
h2{{font-family:Oswald,'Arial Narrow',Impact,sans-serif;font-weight:600;font-size:20px;
 margin:46px 0 4px;color:{NAVY};text-transform:uppercase;letter-spacing:.02em}}
.sub{{color:{MUTED};font-size:14px;margin:0 0 4px}}
.rule{{height:4px;width:64px;background:{YELLOW};margin:14px 0 0}}
.key{{border-left:4px solid {YELLOW};background:#fbfaf4;padding:14px 18px;margin:22px 0 0;
 font-size:16px;max-width:820px}}
table{{border-collapse:collapse;margin:16px 0 0;font-size:14px;font-variant-numeric:tabular-nums}}
th,td{{padding:9px 14px;text-align:right;border-bottom:1px solid #e8e8e8}}
th:first-child,td:first-child{{text-align:left}}
thead th{{font-family:Oswald,'Arial Narrow',sans-serif;font-weight:500;font-size:12.5px;
 text-transform:uppercase;letter-spacing:.04em;color:{MUTED};border-bottom:2px solid {NAVY};
 vertical-align:bottom}}
tbody tr:last-child td{{border-bottom:2px solid #cfcfcf}}
td.n{{color:{MUTED};font-size:12px}}
.best{{background:#f2f7fb}}
.scroll{{overflow-x:auto;max-width:100%}}
.legend{{display:flex;gap:22px;flex-wrap:wrap;margin:18px 0 0;font-size:13px;color:{MUTED}}}
.legend span{{display:inline-flex;align-items:center;gap:7px}}
.sw{{width:13px;height:13px;border-radius:3px;display:inline-block}}
.foot{{margin-top:52px;padding-top:18px;border-top:1px solid #e0e0e0;font-size:12.5px;
 color:{MUTED};max-width:900px}}
.foot li{{margin:5px 0}}
.tag{{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:{MUTED};
 font-family:Oswald,'Arial Narrow',sans-serif}}
.grid2{{display:flex;gap:40px;flex-wrap:wrap;margin-top:14px}}
table.ftab{{margin-top:0;font-size:13.5px}}
table.ftab td.x{{color:#c9c9c9}}
table.ftab .arw{{color:{GOOD_HEX};font-weight:700}}
table.ftab .arw.amb{{color:{MUTED};font-weight:600;cursor:help}}
/* "i" affordance carrying a native title tooltip: recessive enough not to compete with the
   metric name, but a real hit target and cursor change so it reads as hoverable. */
table.ftab .info{{display:inline-flex;align-items:center;justify-content:center;
 width:14px;height:14px;margin-left:6px;border-radius:50%;border:1px solid #c2ccd4;
 color:{MUTED};font-size:9.5px;font-weight:700;font-style:normal;cursor:help;
 vertical-align:middle;line-height:1}}
table.ftab .info:hover{{border-color:{NAVY};color:{NAVY}}}
/* Section 1 layout: scatter fixed-width, feature list takes the remaining space and
   drops below the scatter on narrow viewports rather than squeezing either one. */
.row1{{display:flex;gap:32px;flex-wrap:wrap;align-items:flex-start;margin-top:14px}}
.row1 .scat{{flex:0 0 430px;max-width:430px}}
.row1 .flist{{flex:1 1 320px;min-width:260px}}
.flist .flist-cols{{display:flex;gap:20px;flex-wrap:wrap}}
.flist .flist-cols>div{{flex:1 1 200px;min-width:190px}}
.flist ul{{list-style:none;margin:6px 0 0;padding:0;font-size:12.5px;line-height:1.4}}
.flist li{{padding:5px 0;border-bottom:1px solid #f0f0f0}}
.flist .only{{color:{MUTED};font-size:11px}}
/* Section 3: zone picture beside the band table, wrapping below it on narrow viewports
   rather than crushing the map. The map is a fixed 300px so its cells stay square. */
.zrow{{align-items:flex-start;gap:34px}}
.zrow>svg{{flex:0 0 300px}}
.zrow>.scroll{{flex:1 1 300px;min-width:280px}}
table.ztab td.num,table.ztab th.num{{text-align:right}}
table.ztab td.num{{font-variant-numeric:tabular-nums}}
table.hmt{{margin-top:10px}}
table.hmt td.hm{{text-align:center;padding:14px 20px;border:2px solid #fff;min-width:104px}}
table.hmt .hv{{display:block;font-size:15px;font-weight:600;font-variant-numeric:tabular-nums}}
table.hmt .hn{{display:block;font-size:11px;margin-top:2px}}
table.hmt th.rh{{text-align:left;font-family:Oswald,'Arial Narrow',sans-serif;
 font-weight:500;font-size:12.5px;text-transform:uppercase;letter-spacing:.03em;
 color:{MUTED};border-bottom:0;white-space:nowrap}}
table.hmt thead th{{text-align:center;border-bottom:0}}
table.hmt td.corner{{font-size:11.5px;color:{MUTED};text-align:left;border-bottom:0;
 font-family:Oswald,'Arial Narrow',sans-serif;text-transform:uppercase;
 letter-spacing:.03em}}
/* Axis titles sit outside the band labels: column title centred above them, row title
   rotated to the left of them. Both are headings, so they take the display face and the
   darker navy rather than the muted label grey. */
table.hmt th.axtop{{text-align:center;border-bottom:0;padding:0 14px 5px;
 font-family:Oswald,'Arial Narrow',sans-serif;font-weight:600;font-size:13px;
 text-transform:uppercase;letter-spacing:.04em;color:{NAVY}}}
table.hmt th.axleft{{border-bottom:0;padding:0 8px 0 0;vertical-align:middle;
 text-align:center;width:22px}}
table.hmt th.axleft span{{display:inline-block;writing-mode:vertical-rl;
 transform:rotate(180deg);font-family:Oswald,'Arial Narrow',sans-serif;font-weight:600;
 font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:{NAVY}}}
"""


def bar_chart(items, unit, width=780, row_h=42, pad_l=196):
    """Horizontal bars with a bootstrap uncertainty whisker on each.

    `items` is [(label, value, se, color)]. Value is the worst-third-minus-best-third
    spread, so bigger = sorts better and a bar whose whisker reaches zero is a measure
    that may not sort at all. The x range is extended to include any negative whisker
    end rather than clipping it, because clipping would hide exactly that case.

    The whiskers are per-column and therefore OVERSTATE the uncertainty of a DIFFERENCE
    between two bars: the two spreads move together across resamples. Overlapping
    whiskers do not prove a tie on their own, so the caption carries the paired result.
    """
    lo_v = min(0.0, min(v - s for _, v, s, _ in items))
    hi_v = max(v + s for _, v, s, _ in items) or 1.0
    span = (hi_v - lo_v) or 1.0
    inner = width - pad_l - 86
    h = row_h * len(items) + 14

    def x(v):
        return pad_l + inner * (v - lo_v) / span

    out = [f'<svg viewBox="0 0 {width} {h}" width="100%" height="{h}" '
           f'role="img" aria-label="Sorting power in {unit}, with uncertainty">']
    x0 = x(0.0)
    if lo_v < 0:
        out.append(f'<line x1="{x0:.1f}" y1="4" x2="{x0:.1f}" y2="{h - 8}" '
                   f'stroke="{GRAY}" stroke-width="1" stroke-dasharray="3 3"/>')
    for i, (label, val, se, color) in enumerate(items):
        y = i * row_h + 8
        cy = y + 11
        bx, bw = min(x0, x(val)), abs(x(val) - x0)
        out.append(f'<text x="{pad_l - 12}" y="{cy + 5}" text-anchor="end" '
                   f'font-size="13.5" fill="{INK}">{label}</text>')
        out.append(f'<rect x="{bx:.1f}" y="{y + 3}" width="{max(2.0, bw):.1f}" '
                   f'height="16" rx="4" fill="{color}"/>')
        a, b = x(val - se), x(val + se)
        out.append(f'<line x1="{a:.1f}" y1="{cy}" x2="{b:.1f}" y2="{cy}" '
                   f'stroke="{NAVY}" stroke-width="2"/>')
        for cx in (a, b):
            out.append(f'<line x1="{cx:.1f}" y1="{cy - 5}" x2="{cx:.1f}" y2="{cy + 5}" '
                       f'stroke="{NAVY}" stroke-width="2"/>')
        out.append(f'<text x="{b + 10:.1f}" y="{cy + 5}" font-size="13.5" '
                   f'font-weight="600" fill="{INK}">{val:+.2f}</text>')
    out.append("</svg>")
    return "".join(out)


GOOD, BAD, MID = (26, 127, 79), (179, 38, 30), (247, 247, 247)


def _blend(rgb, t):
    """Interpolate from the neutral midpoint toward `rgb` by t in [0,1]."""
    return tuple(round(MID[i] + (rgb[i] - MID[i]) * t) for i in range(3))


def zone_map(lm, width=300, height=360):
    """The Location+ model, drawn. This is the analog of Section 1's metric table.

    Stuff+ is a ridge, so "what it values" is a list of features and weights. Location+ has
    exactly ONE input -- where the pitch crosses the plate, binned at 0.25 ft -- so there is
    no coefficient table to print. The map IS the model, completely specified, and a picture
    of it is strictly more informative than any table of it would be.

    ORIENTATION, established from data rather than from convention (the project has shipped
    one inverted interpretation already): positive PlateLocSide is the RIGHT-handed batter's
    side. Evidence is hit-by-pitch location, where the ball is at the batter's body: mean
    PlateLocSide is +1.73 ft for RHB and -1.79 ft for LHB over 25,157 HBP. So the panel
    labels its two sides by BATTER HAND, which is unambiguous, instead of asserting a
    catcher's-eye or pitcher's-eye view that a reader could resolve either way.

    Deliberately NOT labelled inside/away HERE, though that labelling is legitimate: the
    VALUE model is pooled across batter handedness, so a given x carries one number whether
    it was inside to a righty or away from a lefty. The coach dashboard's per-pitcher table
    does name regions in/away, and correctly so -- it mirrors PlateLocSide by batter hand to
    BIN pitches, then averages this same pooled map inside each bin (arsenal.py
    _side_relative / _region_label). That is re-binning, not a platoon claim. This panel is
    a picture of the map itself rather than of one pitcher's bins, and at that level the
    axis is absolute, so it is labelled by batter side instead.

    Verified 2026-08-17 that the dashboard's Location+ and this one are the SAME model:
    14_pitcher_pages.py builds its grade from fc.PooledLocationMap, not CountLocationMap,
    so both artifacts in front of the coach share one location-value model. The dashboard
    adds region binning and an occupancy/placement split on top; its count breakdown is
    frequency-only and carries no points, because the map is count-blind.

    KNOWN COST of the pooling, measured, not assumed: fitting the map separately per batter
    hand gives cell values that correlate only +0.73 across hands, mean absolute difference
    3.8 runs/100 against a map spread of roughly 23. So batter-hand mirroring of the VALUE
    map is a real candidate improvement, and it is the same class of bug as the pitcher-hand
    mirroring already adopted for RelSide/HorzBreak. It changes Location+, so it needs the
    pre-registered gate rather than an edit here.

    Colour: expected runs, LOWER = BETTER, so negative cells shade GREEN and positive RED,
    diverging around zero, the same ramp and midpoint as heatmap() above. Colour is not the
    only encoding -- every cell carries a native SVG <title>, so hovering any cell reports
    its exact value and sample size. <title> is used on purpose instead of a JS tooltip:
    scripts are silently stripped when the page is opened over file://, and this page gets
    opened from a Downloads copy. A native tooltip survives that.
    """
    cells = lm["cells"]
    z = lm["zone"]
    vmax = max(abs(c["v"]) for c in cells) or 1.0
    x0, x1, z0, z1 = -2.0, 2.0, 0.5, 4.5     # drawn window, feet
    pad_l, pad_b, pad_t = 30, 26, 10
    sx = (width - pad_l - 8) / (x1 - x0)
    sy = (height - pad_b - pad_t) / (z1 - z0)

    def px(x):
        return pad_l + (x - x0) * sx

    def py(zz):
        return height - pad_b - (zz - z0) * sy      # z up

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
           f'style="max-width:{width}px" role="img" aria-label="Run value by pitch '
           f'location, green is better for the pitcher">']
    step = 0.25
    for c in cells:
        if not (x0 <= c["x"] < x1 and z0 <= c["z"] < z1):
            continue
        t = min(1.0, abs(c["v"]) / vmax)
        rgb = _blend(GOOD if c["v"] < 0 else BAD, t)
        side = "RHB" if c["x"] >= 0 else "LHB"
        out.append(
            f'<rect x="{px(c["x"]):.1f}" y="{py(c["z"] + step):.1f}" '
            f'width="{step * sx:.1f}" height="{step * sy:.1f}" fill="rgb{rgb}" '
            f'stroke="#fff" stroke-width="0.4">'
            f'<title>{abs(c["x"]):.2f} ft to the {side} side, {c["z"]:.2f} ft up '
            f'&#183; {c["v"]:+.1f} runs per 100 &#183; {c["n"]:,} pitches</title></rect>')
    # strike zone
    out.append(f'<rect x="{px(z["x0"]):.1f}" y="{py(z["z1"]):.1f}" '
               f'width="{(z["x1"] - z["x0"]) * sx:.1f}" '
               f'height="{(z["z1"] - z["z0"]) * sy:.1f}" fill="none" stroke="{NAVY}" '
               f'stroke-width="2"/>')
    # batter-side labels: the verified orientation, stated on the picture itself
    out.append(f'<text x="{px(1.35):.0f}" y="{height - 9}" font-size="11" fill="{MUTED}" '
               f'text-anchor="middle">RHB side</text>')
    out.append(f'<text x="{px(-1.35):.0f}" y="{height - 9}" font-size="11" fill="{MUTED}" '
               f'text-anchor="middle">LHB side</text>')
    for zz in (2.0, 3.0, 4.0):
        out.append(f'<text x="{pad_l - 5}" y="{py(zz) + 4:.0f}" font-size="10" '
                   f'fill="{MUTED}" text-anchor="end">{zz:.0f}\'</text>')
    out.append("</svg>")
    return "".join(out)


def zone_bands(lm):
    """League usage and value for five distance bands, batter-agnostic by construction.

    Bands are defined on distance from the zone, not on named in/away spots, for the same
    reason zone_map() avoids inside/away: this model is pooled across batter hand. A band
    table stays true at that resolution, and it carries the finding the map shows -- the
    edges are worth more than the heart, and the real cost is missing badly, not being
    hittable.
    """
    vmax = max(abs(b["v100"]) for b in lm["bands"]) or 1.0
    rows = []
    for b in lm["bands"]:
        t = min(1.0, abs(b["v100"]) / vmax)
        rgb = _blend(GOOD if b["v100"] < 0 else BAD, t)
        fg = "#fff" if t > 0.58 else INK
        rows.append(
            f'<tr><td>{b["band"]}</td>'
            f'<td class="num">{b["pct"]:.0f}%</td>'
            f'<td class="num" style="background:rgb{rgb};color:{fg}">'
            f'{b["v100"]:+.2f}</td></tr>')
    return (f'<div class="scroll"><table class="ftab ztab"><thead><tr><th>Where it goes</th>'
            f'<th class="num">Share of pitches</th><th class="num">Runs per 100</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def heatmap(g, value_col, row_label, col_label, vmax=None):
    """3x3 grid, rows = `row_label` score tercile, cols = Location+ tercile.

    Criterion is expected runs, LOWER = BETTER, so negative cells shade GREEN and
    positive shade RED, diverging around zero (an average pitcher). Best third is
    listed FIRST on both axes so the best cell sits top-left, where it is read first.

    Every cell prints its own number, so colour is never the only encoding -- which is
    what keeps a red/green ramp legible to a red-green colourblind reader. Scale is
    shared across both grids on the page when `vmax` is passed, so the two are directly
    comparable rather than each being normalised to its own extremes.
    """
    order = ["best third", "middle", "worst third"]
    vals = [g["cells"][f"{r}|{c}"][value_col] for r in order for c in order
            if g["cells"][f"{r}|{c}"][value_col] is not None]
    vmax = vmax or (max(abs(v) for v in vals) or 1.0)

    head = "".join(f'<th>{c}</th>' for c in order)
    body = []
    for r in order:
        tds = []
        for c in order:
            cell = g["cells"][f"{r}|{c}"]
            v = cell[value_col]
            if v is None:
                tds.append('<td class="hm">&mdash;</td>')
                continue
            t = min(1.0, abs(v) / vmax)
            rgb = _blend(GOOD if v < 0 else BAD, t)
            fg = "#fff" if t > 0.58 else INK
            mark = "*" if cell["thin"] else ""
            tds.append(f'<td class="hm" style="background:rgb{rgb};color:{fg}">'
                       f'<span class="hv">{v:+.2f}</span>'
                       f'<span class="hn" style="color:{fg};opacity:.72">'
                       f'{cell["n"]}{mark}</span></td>')
        # The row-axis title is a single rotated cell spanning every body row, emitted on
        # the first row only. Axis titles sit OUTSIDE the tercile labels -- "Location+"
        # above the column labels, "Stuff+" left of the row labels -- rather than sharing
        # one corner cell, so each title reads as a heading for the band names beneath it
        # instead of as a legend the reader has to decode before the grid makes sense.
        lead = (f'<th class="axleft" rowspan="{len(order)}">'
                f'<span>{row_label}</span></th>') if r == order[0] else ""
        body.append(f'<tr>{lead}<th class="rh">{r}</th>{"".join(tds)}</tr>')
    return (f'<div class="scroll"><table class="hmt">'
            f'<thead>'
            f'<tr><td class="corner" colspan="2"></td>'
            f'<th class="axtop" colspan="{len(order)}">{col_label}</th></tr>'
            f'<tr><td class="corner" colspan="2"></td>{head}</tr>'
            f'</thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


# Four series on one chart, so hue is spread rather than kept inside the brand blues:
# three blues were indistinguishable at 2px. Amber carries Location+, and run value
# stays deliberately neutral as the reference series. Checked with the dataviz validator
# (light surface): CVD separation passes at worst-adjacent dE 13.5 protan / 17.7 normal.
# Two flags are known and accepted -- the grey trips the chroma floor because it is meant
# to read as neutral, and #00a0df sits at 2.88:1 contrast, which the direct label on every
# line discharges. Dash patterns are redundant encoding so the lines survive greyscale.
CURVE_STYLE = {
    "blend":   ("#00539f", 3.0, "",     "Stuff+ and Location+"),
    "stuff":   ("#00a0df", 2.0, "",     "Stuff+"),
    "loc":     ("#c87800", 2.2, "7 4",  "Location+"),
    "results": ("#6f6f6f", 2.0, "2 3",  "2025 Expected Runs Allowed"),
}


# Curves drawn on the sample-size chart, in back-to-front order. The combined
# "Stuff+ and Location+" line is deliberately OMITTED: it tracks Location+ closely at
# high volume (the two converge because independent signals add in quadrature, so a
# fixed-strength Stuff+ contributes a shrinking absolute amount as Location+ grows),
# and a coach reading four lines where two nearly coincide loses the one comparison
# that matters -- how fast each single grade reaches a full season of results.
# Re-add "blend" here to restore it; the data is still computed and cached.
CURVES = ("results", "loc", "stuff")


def line_chart(cv, width=800, height=340, pad_l=208, pad_b=46, pad_t=14, pad_r=54):
    """Validity against next season vs how many fastballs the grade was built from.

    X is LOG-scaled because the interesting region is 10 to 100 pitches, which a linear
    axis would crush into the left margin. Y is correlation with next season's realized
    run value.

    No uncertainty bands: four overlapping ribbons were less legible than the lines they
    described. The SE range lives in the caption instead.
    """
    grid, ref = cv["grid"], cv["reference"]
    lo_x, hi_x, hi_y = math.log(grid[0]), math.log(grid[-1]), 0.20
    iw, ih = width - pad_l - pad_r, height - pad_b - pad_t

    def x(n):
        return pad_l + iw * (math.log(n) - lo_x) / (hi_x - lo_x)

    def y(v):
        return pad_t + ih * (1 - min(hi_y, max(0.0, v)) / hi_y)

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
           f'role="img" aria-label="Predictive validity versus sample size">']
    # y numbers sit on the RIGHT: the left margin is given over entirely to series labels,
    # which are placed at the start of each line rather than the end (the curves are far
    # better separated at 10 pitches than at 250, where two pairs nearly coincide).
    for gv in (0.05, 0.10, 0.15, 0.20):
        out.append(f'<line x1="{pad_l}" y1="{y(gv):.1f}" x2="{pad_l + iw}" '
                   f'y2="{y(gv):.1f}" stroke="#eee" stroke-width="1"/>')
        out.append(f'<text x="{pad_l + iw + 9}" y="{y(gv) + 4:.1f}" '
                   f'font-size="11.5" fill="{MUTED}">{gv:.2f}</text>')
    for n in (10, 20, 50, 100, 250):
        out.append(f'<text x="{x(n):.1f}" y="{height - pad_b + 20}" text-anchor="middle" '
                   f'font-size="11.5" fill="{MUTED}">{n}</text>')
    out.append(f'<text x="{pad_l + iw / 2:.1f}" y="{height - 6}" text-anchor="middle" '
               f'font-size="12" fill="{MUTED}">2025 fastballs the measure is built from'
               f'</text>')
    ymid = pad_t + ih / 2
    out.append(f'<text x="17" y="{ymid:.1f}" font-size="12" fill="{MUTED}" '
               f'text-anchor="middle" transform="rotate(-90 17 {ymid:.1f})">'
               f'2025 correlation with 2026 expected runs &rarr;</text>')

    for key in CURVES:
        c = cv["curves"][key]
        col, wdt, dash, _ = CURVE_STYLE[key]
        pts = " ".join(f"{x(n):.1f},{y(r):.1f}" for n, r in zip(grid, c["r"]))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" '
                   f'stroke-width="{wdt}" stroke-dasharray="{dash}" '
                   f'stroke-linejoin="round"/>')
        out.append(f'<circle cx="{x(grid[0]):.1f}" cy="{y(c["r"][0]):.1f}" r="4" '
                   f'fill="{col}"/>')

    # Left-margin labels, anchored to where each line STARTS. Even here two can coincide
    # (the blend opens at 0.100 against the reference at 0.102), so they are separated
    # greedily and any label that had to move draws a leader back to its own line.
    lx = x(grid[0]) - 12
    labels = sorted((y(cv["curves"][k]["r"][0]), CURVE_STYLE[k][3], CURVE_STYLE[k][0])
                    for k in CURVES)
    placed, prev = [], -1e9
    for anchor, text, col in labels:
        pos = max(anchor, prev + 17)
        placed.append((anchor, pos, text, col))
        prev = pos
    shift = max(0.0, placed[-1][1] - (height - pad_b - 4))
    for anchor, pos, text, col in placed:
        pos -= shift
        if abs(pos - anchor) > 1.5:      # nudged: connect it back to its own line
            out.append(f'<path d="M {lx + 4:.1f} {pos:.1f} L {lx + 9:.1f} {anchor:.1f}" '
                       f'stroke="{col}" stroke-width="1" fill="none" opacity="0.55"/>')
        out.append(f'<text x="{lx:.1f}" y="{pos + 4:.1f}" text-anchor="end" '
                   f'font-size="12.5" font-weight="600" fill="{INK}">{text}</text>')
    out.append("</svg>")
    return "".join(out)


def scatter(xs, ys, r, width=430, height=430, pad=54):
    """v1 against v2 on the shared 100 +/- 15 display scale. One dot per pitcher, no
    identifiers of any kind. The dashed diagonal is exact agreement."""
    lo, hi = 55, 145

    def p(v):
        return pad + (width - 2 * pad) * (min(hi, max(lo, v)) - lo) / (hi - lo)

    def q(v):
        return height - pad - (height - 2 * pad) * (min(hi, max(lo, v)) - lo) / (hi - lo)

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
           f'style="max-width:{width}px" role="img" '
           f'aria-label="v1 grade against v2 Stuff+ grade, correlation {r:.2f}">']
    out.append(f'<rect x="{pad}" y="{pad}" width="{width - 2 * pad}" '
               f'height="{height - 2 * pad}" fill="none" stroke="#ececec"/>')
    out.append(f'<line x1="{p(lo)}" y1="{q(lo)}" x2="{p(hi)}" y2="{q(hi)}" '
               f'stroke="{GRAY}" stroke-width="1" stroke-dasharray="4 4"/>')
    for a, b in zip(xs, ys):
        out.append(f'<circle cx="{p(a):.1f}" cy="{q(b):.1f}" r="2.6" fill="{BLUE}" '
                   f'opacity="0.30"/>')
    out.append(f'<text x="{width / 2:.0f}" y="{height - 12}" text-anchor="middle" '
               f'font-size="12" fill="{MUTED}">v1 Stuff+ grade &rarr;</text>')
    out.append(f'<text x="17" y="{height / 2:.0f}" font-size="12" fill="{MUTED}" '
               f'transform="rotate(-90 17 {height / 2:.0f})" '
               f'text-anchor="middle">v2 Stuff+ grade &rarr;</text>')
    out.append(f'<text x="{pad + 12}" y="{pad + 26}" font-size="22" font-weight="600" '
               f'fill="{NAVY}" font-family="Oswald,sans-serif">r = {r:.2f}</text>')
    out.append("</svg>")
    return "".join(out)


def thirds_table(cols, table, unit):
    hdr = "".join(f"<th>{c['label']}</th>" for c in cols)
    body = []
    for band in BANDS:
        tds = []
        for c in cols:
            cell = table[c["key"]]["bands"][band]
            klass = ' class="best"' if band == "best third" else ""
            tds.append(f'<td{klass}>{cell["mean"]:+.2f} '
                       f'<span class="n">&plusmn;{cell["se"]:.2f}</span></td>')
        body.append(f'<tr><td>{BAND_NOTE[band]}</td>{"".join(tds)}</tr>')
    counts = "".join(f'<td class="n">{table[c["key"]]["bands"]["best third"]["n"]} '
                     f'pitchers/third</td>' for c in cols)
    spreads = "".join(f'<td><strong>{table[c["key"]]["spread"]:+.2f}</strong></td>'
                      for c in cols)
    return (f'<div class="scroll"><table><thead><tr><th>2025 grade</th>{hdr}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody>'
            f'<tfoot><tr><td class="tag">worst minus best</td>{spreads}</tr>'
            f'<tr><td class="n"></td>{counts}</tr></tfoot></table></div>')


def feature_importance_panel(fi):
    """v1 and v2 input lists, ranked by each model's own |correlation with its own
    score|, so the coach can see the model reads the pitch through mostly the same
    lens he does. Rows the coach's card does not use at all are flagged inline rather
    than in a separate legend, since there are few enough to read in place."""
    label_of = {r["feature"]: r["label"] for r in fi["rows"]}

    def direction(better_when):
        return "more is better" if better_when == "more" else "less is better"

    # Reframe for a feature whose ISOLATED coefficient reads backwards as a suppression
    # artifact. Shown only while the feature is still in that "less is better" state, so it
    # SELF-SUPPRESSES if the fix is adopted (e.g. Extension flips to "more is better" once
    # the ridge is refit on RelSpeed). Keyed by the analysis feature name, not the label.
    # This is the fallback that keeps the extension row honest even if the refit does not
    # land -- README open item #1 -- and needs only a re-render, no data rebuild.
    REFRAME = {
        "Extension": ("reads backwards only in isolation &mdash; more extension goes with a "
                      "better grade overall; effective velo already banks its benefit, so "
                      "the leftover term prices &lsquo;less arm speed for the same perceived "
                      "velo&rsquo;"),
    }

    # One row per metric, one column per model: a number where the model uses it, an x
    # where it does not. The arrow carries direction and sits in the CELL, not the row,
    # because the two models disagree on some (horizontal break reads up for v1 and down
    # for ours) -- putting it per-row would hide exactly the disagreement worth seeing.
    # Keyed by LABEL, not feature name: the two models reach the same physical metric
    # through different columns (his raw HorzBreak/RelSide vs our arm-side mirrors). Keying
    # by feature name split those into duplicate rows with an x in one model each, which
    # read as "his card has no break term" -- exactly wrong. Velocity is deliberately NOT
    # merged this way; see the PRETTY comment in coach_feature_importance.py.
    #
    # OFF THE PAGE, STILL IN THE MODEL. Every grade on this page comes from a ridge that
    # trains on all 12 FEATS including these four. Dropping a feature changes ridge_pred and
    # so requires re-running the adoption gate; that is queued as post-meeting work, not
    # done here. What is suppressed is the metric-table ROW, because each of these invites a
    # coach question that the model cannot honestly answer. Two distinct reasons:
    #
    # (a) degenerate on a four-seam-only model -- the three "vs his fastball" differentials
    #     are computed against Avg_*_FastestType, and on a FOUR-SEAM the pitcher's fastest
    #     type IS his four-seam for nearly every pitcher. The feature reduces to this pitch
    #     minus his own four-seam average: within-pitcher scatter, averaging to ~0 per
    #     pitcher by construction. Measured on the D1 cache, means -0.07/+0.12/-0.06 over
    #     1.35M four-seams; importances 0.03, 0.00, 0.05, the three smallest in the model.
    #     They become meaningful only in a multi-pitch model, against a real second pitch.
    #
    # (b) wrong construct -- "Batter hits left" is a real platoon effect, not a degenerate
    #     one (importance 0.04), but it is opponent context, not pitch quality. Leaving it
    #     in a STUFF score credits a pitcher for the batters he happened to face, which is
    #     the occupancy leak the project's modeling principles forbid: condition context at
    #     the pitch level, subtract it at the pitcher level. The post-meeting fix is to
    #     neutralize it at aggregation (score every pitcher against a common batter mix)
    #     rather than delete it, so genuine lefty-specialist usage stays visible as its own
    #     labeled component instead of hiding inside Stuff+.
    OFF_PAGE = {"Velo vs his own fastball", "Vert break vs his fastball",
                "Horz break vs his fastball", "Batter hits left"}
    by = {}
    for r in fi["rows"]:
        if r["label"] in OFF_PAGE:
            continue
        by.setdefault(r["label"], {})[r["model"]] = r

    def cell(r):
        if r is None:
            return '<td class="x">&times;</td>'
        # "magnitude": his off-average terms reward distance from the hand-typical value in
        # EITHER direction, so no up/down arrow is truthful. Shown as a muted plus-minus.
        if r["better_when"] == "magnitude":
            return ('<td>{:.2f} <span class="arw amb" title="Scored as distance from '
                    'the typical release point, so either direction earns points">'
                    '&plusmn;</span></td>').format(abs(r["r_with_own_score"]))
        arrow = "&uarr;" if r["better_when"] == "more" else "&darr;"
        return (f'<td>{abs(r["r_with_own_score"]):.2f} '
                f'<span class="arw">{arrow}</span></td>')

    # Sorted by how much each metric drives OUR grade, descending. Previously this sorted on
    # the max across both models, which let a metric his card leans on outrank one ours does
    # and made the column read as if it were ranked by nothing in particular. Anything our
    # model does not use has no v2 value and sinks to the bottom (ties broken by his own
    # importance), so his unique input lands last rather than being dropped.
    # A metric and its "vs typical" partner are kept ADJACENT rather than sorted apart. They
    # are two readings of the same physical measurement -- feet toward the arm side versus
    # distance from a typical release point -- and the comparison a coach wants is between
    # those two readings, which only works if the rows touch. The pair sits at the position
    # of its stronger member, and within the pair the plain reading leads.
    def imp(row, model):
        return abs(row[model]["r_with_own_score"]) if model in row else -1.0

    # Extension is PINNED directly under Velocity, above metrics that outrank it. That is a
    # narrative adjacency, not an importance claim: his card's single velocity term is
    # effective velo, which is computed FROM release speed and extension, so ours splits into
    # exactly these two rows. Read together they show the split; ranked apart they don't.
    # Effective velocity, Velocity, Extension are one cluster in that fixed order, and it
    # lands first because Velocity carries the largest v2 importance. The three rows read as
    # a sentence: his card has one bundled velocity term, ours splits it into release speed
    # and extension. RANK overrides importance inside the cluster, deliberately.
    CLUSTER = {"Extension": "Velocity", "Effective velocity": "Velocity"}
    RANK = {"Effective velocity": 0, "Velocity": 1, "Extension": 2}

    groups = {}
    for label, m in by.items():
        base = label.replace(" vs typical", "")
        groups.setdefault(CLUSTER.get(base, base), []).append((base, label, m))

    def within(member):
        base, label, m = member
        return (RANK.get(base, 99),                            # fixed order where declared
                0 if CLUSTER.get(base, base) == base else 1,   # otherwise cluster head leads
                0 if label == base else 1,                     # plain before "vs typical"
                -imp(m, "v2"), -imp(m, "v1"))

    order = []
    for _, members in sorted(
            groups.items(),
            key=lambda kv: (-max(imp(m, "v2") for _, _, m in kv[1]),
                            -max(imp(m, "v1") for _, _, m in kv[1]))):
        order.extend((label, m) for _, label, m in sorted(members, key=within))
    # The arm-side frame is explained on an "i" beside the metric it applies to, rather than
    # as a paragraph under the table. An up-arrow on a row called "Release side" otherwise
    # reads as "a bigger number in feet", which is only true for a righty: our feature is
    # signed toward the throwing arm, so "more" means a MORE POSITIVE value for a RHP and a
    # MORE NEGATIVE one for a LHP. Delivered as a title attribute, not a JS tooltip, so it
    # survives being opened over file:// where scripts are stripped.
    ARM_NOTE = ("Measured toward the throwing arm, so &uarr; means further to the arm "
                "side: a higher number for a righty, a lower one for a lefty.")

    def typical_note(m):
        """Tooltip for a 'vs typical' row: what 'typical' is measured against.

        The distinction worth killing on sight is against the LEAGUE versus against the
        pitcher's OWN average. These terms centre on the per-hand mean over four-seams in
        the training year, i.e. the average pitcher of that handedness -- not the pitcher's
        own norm, which is what the (dropped) 'vs his fastball' differentials did. Centres
        are read from fc.DEV_CENTRES so the numbers here cannot drift from the model's.
        """
        feat = str(m.get("v2", {}).get("feature", ""))
        c = fc.DEV_CENTRES.get(feat)
        if not c:
            return None
        frame = (", measured toward the throwing arm"
                 if str(fc.DEV_SRC.get(feat, "")).endswith("_arm") else "")
        return (f"How far from the average four-seam release point for a pitcher of the "
                f"same hand ({c[0]:.2f} ft for righties, {c[1]:.2f} ft for lefties"
                f"{frame}), counted in either direction: unusually high and unusually low "
                f"both earn credit. Not measured against the pitcher's own average. Each "
                f"card uses its own per-hand figure; they differ by hundredths of a foot.")

    def name_cell(label, m):
        arm = "v2" in m and str(m["v2"]["feature"]).endswith("_arm")
        note = ARM_NOTE if arm else typical_note(m)
        info = f'<span class="info" title="{note}">i</span>' if note else ""
        return f'<td>{label}{info}</td>'

    body = "".join(
        f'<tr>{name_cell(f, m)}{cell(m.get("v1"))}{cell(m.get("v2"))}</tr>'
        for f, m in order)
    table = (f'<div class="scroll"><table class="ftab"><thead><tr><th>Metric</th>'
             f'<th>v1 Stuff+</th><th>v2 Stuff+</th></tr></thead>'
             f'<tbody>{body}</tbody></table></div>')

    notes = "".join(
        f'<p class="sub"><strong>{f}</strong> {REFRAME[f]}.</p>'
        for f, m in by.items()
        if f in REFRAME and any(v["better_when"] == "less" for v in m.values()))

    return f'<div class="flist">{table}{notes}</div>'


def main() -> int:
    args = fc.paths()
    with open(os.path.join(args.workdir, "coach_page_data.json")) as fh:
        d = json.load(fh)
    with open(os.path.join(args.workdir, "coach_sample_curve.json")) as fh:
        cv = json.load(fh)
    with open(os.path.join(args.workdir, "coach_feature_importance.json")) as fh:
        fi = json.load(fh)
    # Location+ map: one input, so its "importance" is the map itself (see zone_map).
    with open(os.path.join(args.workdir, "coach_location_map.json")) as fh:
        lm = json.load(fh)

    # agreement panel, straight off the cached pool: no rebuild, no names, two derived
    # grades per pitcher put on the shared 100 +/- 15 display scale
    pool = pd.read_parquet(os.path.join(args.workdir, "coach_compare_pool.parquet"))
    ag_x = 100 + 15 * cp.z(pool["coach_hi"].to_numpy())
    ag_y = 100 + 15 * cp.z(pool["ours_hi"].to_numpy())
    ag_r = float(fc.R(pd.Series(ag_x), pd.Series(ag_y)))

    # point-estimate crossing only. The paired test has no power at these SEs -- it called
    # every measure a match at 10 pitches, including results against themselves -- so it
    # is deliberately NOT reported as significance.
    def crosses(key):
        return next((n for n, r in zip(cv["grid"], cv["curves"][key]["r"])
                     if r >= cv["reference"]["r"]), None)

    # Apply the coach-facing labels over whatever the analysis JSON carried.
    cols = [dict(c, label=LABELS.get(c["key"], c["label"])) for c in d["columns"]]
    missing = [c["key"] for c in d["columns"] if c["key"] not in LABELS]
    if missing:
        raise SystemExit(f"no coach-facing label for {missing}; add it to LABELS rather "
                         f"than letting an analysis label reach the page")
    ff = d["ff"]["100"]
    tgt = ff["by_criterion"]["Target"]
    adj = ff["by_criterion"]["adjT"]

    def items_for(entry, use_cols):
        """Bar length is the POINT spread (so it matches the table exactly); the whisker
        is the bootstrap SE of that spread."""
        return [(c["label"], entry["table"][c["key"]]["spread"],
                 entry["spread_boot"][c["key"]]["se"], COLOR[c["key"]]) for c in use_cols]

    # Section 4 drops v1 entirely once the comparison has landed in Section 1.
    _order4 = ["velo_hi", "prior_hi", "loc_hi", "stuff_hi", "pitch2_hi"]
    _by = {c["key"]: c for c in cols}
    cols4 = [_by[k] for k in _order4 if k in _by]
    tgt_items, adj_items = items_for(tgt, cols4), items_for(adj, cols4)

    # Section 1's own two-bar chart: v1 against v2 only, luck-adjusted only. This is the
    # ONE place ours is called "v2 Stuff+" -- the comparison is the point here, so the
    # version number earns its keep; everywhere after this it is just "Stuff+".
    compare_cols = [dict(c, label="v2 Stuff+" if c["key"] == "stuff_hi" else c["label"])
                    for c in cols if c["key"] in ("coach_hi", "stuff_hi")]
    adj_two = items_for(adj, compare_cols)

    # v1-vs-v2 paired difference, from the earlier run over the same pool. Selected by
    # matching pool size, not by key name: that file's "primary"/"sensitivity" keys do
    # not reliably indicate which floor produced them.
    with open(os.path.join(args.workdir, "coach_ff_criterion.json")) as fh:
        prior = json.load(fh)
    block = next((b for b in prior.values() if b.get("n") == ff["n"]), None)
    if block is None:
        raise SystemExit(f"no coach_ff_criterion block with n={ff['n']}; refusing to "
                         f"quote a paired difference from a different pool")
    pair = {v: block["by_criterion"][v]["vs_stuff"]["coach_hi"] for v in ("Target", "adjT")}

    grids = ff["grids"]
    sx = grids["stuff_x_loc"]

    def gmax(vc):
        """Colour scale for the Stuff+ x Location grid, shared between the raw and
        luck-adjusted panes of the SAME criterion so toggling doesn't rescale the ramp."""
        vals = [c[vc] for c in sx["cells"].values() if c[vc] is not None]
        return max(abs(v) for v in vals) or 1.0

    def sort_pane(entry, items, unit, note):
        return (f'{bar_chart(items, unit)}'
                f'<p class="sub">{note}</p>')

    tie_raw = (f"v1 and v2 are a statistical tie here: v1 leads by "
               f"{pair['Target']['mean']:+.2f} runs, with a standard error of "
               f"{pair['Target']['se']:.2f} on that difference.")
    tie_adj = (f"v1 and v2 are a statistical tie here too, with the order reversed: v2 "
               f"leads by {abs(pair['adjT']['mean']):.2f} runs, standard error "
               f"{pair['adjT']['se']:.2f}.")

    vs = tgt["vs_blend"]
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grading Fastballs</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Oswald:wght@500;600&family=Open+Sans:wght@400;600&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body><div class="wrap">
<h1>Grading Fastballs</h1>
<p class="sub">{ff['n']} DI Pitchers graded on their 2025 four-seams, measured by their
2026 four-seam results.</p>
<div class="rule"></div>

<h2>1. Where the two grades already agree</h2>
<div class="row1">
 <div class="scat">{scatter(ag_x, ag_y, ag_r)}</div>
 {feature_importance_panel(fi)}
</div>

<h2>2. Sorting next season's luck-adjusted run value</h2>
{bar_chart(adj_two, "runs per 100 fastballs")}

<h2>3. Location+</h2>
<p class="sub">Run value per 100 four-seams by where the pitch crossed,
<strong>green good, red bad</strong>. Hover any square for its exact value.</p>
<div class="grid2 zrow">{zone_map(lm)}{zone_bands(lm)}</div>

<h2>4. Location+ Impact</h2>
<p class="sub">2026 run value per 100 fastballs, <strong>green good, red bad</strong>,
small number is how many pitchers are in the cell.</p>
{heatmap(sx, "crit100_adjT", "Stuff+", "Location+", gmax("crit100_adjT"))}

<h2>5. Pitches Required</h2>
<p class="sub">How well you can predict following season Expected Runs allowed by how
many pitches are observed. {cv['pool']} pitchers with {cv['n_max']}+ four-seams.</p>
{line_chart(cv)}

<h2>6. All of it side by side</h2>
<p class="sub">Gap between the best-graded third and the worst-graded third, in runs per
100 fastballs. Longer is better. Whiskers are the resampling range on each bar.</p>
{sort_pane(adj, adj_items, "runs per 100 fastballs",
           "Every measure on the page at once, batted-ball luck removed.")}


<div class="foot"><strong>Reading the numbers</strong>
<ul>
<li><strong>Run value per 100 fastballs</strong> is measured against an average pitcher,
so negative is good and zero is average. Each pitch is charged the change in run
expectancy it caused, plus any runs that scored on it. A double with nobody on costs
about six tenths of a run whether or not that runner later scores.</li>
<li><strong>"Expected Runs Allowed" includes batted-ball luck</strong>: a hard-hit ball
caught at the wall counts as an out, and a soft one that finds a hole counts as a hit.
<strong>"Luck-adjusted"</strong> replaces each batted ball with the average run value for
its exit speed and launch angle, so the fielding and the bounces come out. Section 1 shows
one pair of grades whose ordering flips between the two, which is why both are on the page
instead of whichever one we preferred.</li>
<li><strong>Grades are on a 100-average scale</strong> and every column is put on the same
scale, so a "best third" means the same thing across columns. A given pitcher can land in
different thirds in different columns, which is the whole comparison.</li>
<li><strong>"Last year's run value" is not one of our grades.</strong> It is the run value
of the pitcher's own four-seams the previous season, used as the bar a grade has to clear.
It needs a full season of pitches; the grades need about 100.</li>
<li><strong>Whiskers overstate the case for a tie.</strong> They are each bar's own
resampling range, and two bars move together across resamples, so overlap alone does not
establish a tie. The paired figures decide it: on actual results, Stuff+ with Location+
beats Stuff+ alone by {vs['stuff_hi']['mean']:+.2f} runs per 100
(&plusmn;{vs['stuff_hi']['se']:.2f}), and runs {vs['prior_hi']['mean']:+.2f}
(&plusmn;{vs['prior_hi']['se']:.2f}) against a full season of results. That last one does
not clear our significance bar, so read it as "at least as good as", not "better than".</li>
<li><strong>Scope</strong>: innings 1 through 8 only. The upstream build drops ninth-inning
and extra-inning pitches from the graded data, so late-inning relief work is
under-represented and closers most of all. Fastballs only throughout; Location+ is a
fastball-location measure.</li>
</ul></div>
</div></body></html>"""

    dest = os.path.join(args.workdir, "coach_fastball_comparison.html")
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
