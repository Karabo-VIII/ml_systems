"""Render the full TI x TF DNA into a self-contained HTML dashboard (no external deps).

Style follows scripts/autonomy/dashboard.py (inline CSS, color-coded cells). Builds a
heatmap matrix: rows = TI families (grouped MA / oscillator / volume / misc), columns =
ALL 5 timeframes {1d, 4h, 2h, 1h, 30m}. Each cell shows OOS-2024 edge/wk with a VAL
sub-line, colored by edge sign+magnitude, annotated with null pct + deploy + a profile tag.

Merges multiple DNA JSON runs (e.g. the 1d/4h/2h run + the 1h/30m run) so NOTHING is
left behind. Usage:
  python -m strat.ti_dna_html  [json1 json2 ...]   (defaults: newest 2 ti_dna_*.json)
"""
from __future__ import annotations
import sys, json, html
from pathlib import Path
from datetime import datetime

DNA_DIR = Path(__file__).resolve().parents[1].parent / "runs" / "engines" / "V5_dna_report"

FAMILY_GROUPS = {
    "Moving Averages": ["EMA", "SMA", "WMA", "HMA", "DEMA", "TEMA", "KAMA", "VIDYA"],
    "Oscillators / Momentum": ["RSI", "MACD", "BOLLINGER", "STOCHASTIC", "ADX", "SUPERTREND"],
    "Volume (real chimera)": ["OBV", "VWAP", "CMF", "MFI", "FLOW"],
    "Misc / Trend": ["DONCHIAN", "CCI", "WILLR", "PSAR", "AROON", "VORTEX", "KELTNER"],
}
TFS = ["1d", "4h", "2h", "1h", "30m"]


def load_rows(paths):
    merged = {}
    meta = {}
    for p in paths:
        data = json.load(open(p))
        meta = data
        for r in data["rows"]:
            merged[(r["family"], r["tf"])] = r
    return merged, meta


def edge_color(edge, null, deploy):
    """Green = strong+null-confirmed, amber = positive, grey = abstain, red = negative."""
    if deploy is not None and deploy < 1e-3:
        return "#3a3a3a", "#888"          # abstain
    if edge is None:
        return "#2a2a2a", "#888"
    if edge > 0:
        if null is not None and null >= 95:
            # intensity by magnitude
            if edge >= 2.0:
                return "#1b7d3e", "#fff"
            if edge >= 1.0:
                return "#2f9e54", "#fff"
            return "#3cb371", "#fff"
        return "#6b8f3a", "#fff"          # positive but weak null
    else:
        if edge <= -2.0:
            return "#a52828", "#fff"
        if edge <= -0.5:
            return "#c0504d", "#fff"
        return "#8a5a5a", "#eee"          # mildly negative


def profile_tag(r):
    v, o = r.get("VAL"), r.get("OOS2024")
    if not v or not o:
        return "n/a"
    od = o["deploy"]
    if od < 1e-3:
        return "abstain (2h synth)"
    ve, oe = v["edge_pp"], o["edge_pp"]
    vn, on = r["VAL_null_pct"], r["OOS_null_pct"]
    sel = v["deploy"] < 0.70 and od < 0.70
    if sel and ve > 0 and oe > 0 and on >= 95:
        return "SELECTIVE+ROBUST"
    if ve > 0 and oe > 0 and vn >= 95 and on >= 95:
        return "robust tilt"
    if ve > 0 and oe > 0:
        return "consistent"
    if oe > 0 and on >= 95:
        return "OOS-only"
    if ve > 0 and vn >= 95:
        return "VAL-only"
    return "null/bleed"


def cell_html(r):
    if r is None:
        return '<td class="cell na">—</td>'
    v, o = r.get("VAL"), r.get("OOS2024")
    if not v or not o:
        return '<td class="cell na">—</td>'
    oe = o["edge_pp"]; on = r["OOS_null_pct"]; od = o["deploy"]
    bg, fg = edge_color(oe, on, od)
    tag = profile_tag(r)
    if od < 1e-3:
        body = "<b>abstain</b><br><span class=sub>no native-2h base</span>"
    else:
        body = (f"<b>{oe:+.2f}</b> pp/wk<br>"
                f"<span class=sub>null {on:.0f} · dep {od:.0%} · Sh {o['sharpe']:.2f}</span><br>"
                f"<span class=sub>VAL {v['edge_pp']:+.2f} · comp {o['compound']:+.0f}%</span><br>"
                f"<span class=tag>{tag}</span>")
    return f'<td class="cell" style="background:{bg};color:{fg}">{body}</td>'


def build(paths, asset_note):
    merged, meta = load_rows(paths)
    win = meta.get("windows", {})
    val_w = win.get("VAL", ["?", "?"]); oos_w = win.get("OOS2024", ["?", "?"])
    n_null = meta.get("n_null", "?")
    cost = meta.get("cost_rt", "?")

    parts = []
    parts.append(f"""<!doctype html><html><head><meta charset=utf-8>
<title>TI x Timeframe DNA — 2024 OOS</title>
<style>
 body{{background:#161616;color:#ddd;font:13px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px}}
 h1{{font-size:21px;margin:0 0 4px}} h2{{font-size:15px;color:#9cc;margin:26px 0 8px}}
 .meta{{color:#999;font-size:12px;margin-bottom:6px}}
 .legend span{{display:inline-block;padding:2px 8px;margin:2px 4px 2px 0;border-radius:3px;font-size:11px}}
 table{{border-collapse:collapse;margin:6px 0 14px;width:100%}}
 th,td{{border:1px solid #2c2c2c;padding:6px 8px;text-align:center;vertical-align:middle}}
 th{{background:#222;color:#bbb;font-size:12px}}
 th.fam{{text-align:left;min-width:92px;background:#1d1d1d}}
 td.fam{{text-align:left;font-weight:bold;background:#1d1d1d;color:#cde}}
 .cell{{min-width:118px;font-size:11.5px}} .cell.na{{background:#222;color:#666}}
 .sub{{font-size:10px;opacity:.85}} .tag{{font-size:9.5px;opacity:.8;font-style:italic}}
 .note{{background:#202020;border-left:3px solid #c89b3c;padding:10px 14px;margin:14px 0;font-size:12px;color:#ccc}}
 .ok{{border-left-color:#3cb371}}
</style></head><body>""")
    parts.append("<h1>TI &times; Timeframe — Full Performance DNA (2024 OOS)</h1>")
    parts.append(f"<div class=meta>VAL {val_w[0]} &rarr; {val_w[1]} &nbsp;|&nbsp; "
                 f"OOS-2024 {oos_w[0]} &rarr; {oos_w[1]} &nbsp;(<b>2025 sealed</b>) &nbsp;|&nbsp; "
                 f"gate BTC-SMA100 &nbsp;|&nbsp; maker {cost} RT &nbsp;|&nbsp; EW u100 &nbsp;|&nbsp; "
                 f"null draws {n_null} &nbsp;|&nbsp; generated {datetime.now():%Y-%m-%d %H:%M}</div>")

    parts.append("""<div class=note ok><b>Pipeline hyper-audited: 19/19 checks PASSED</b> —
 indicator causality (truncation test), data wall (2025 absent), look-ahead sensitivity
 (cheat inflates +65&rarr;+4217%), cost application, regime gate, random-null construction.
 Numbers are causally clean. See PIPELINE_AUDIT_RESULT.md.</div>""")
    parts.append(f"""<div class=note><b>Universe (point-in-time honesty):</b> {asset_note}
 The engine uses each asset's causal history and only bars inside each window, so newer
 listings contribute zero bars before they existed (no price look-ahead). However the u100
 <i>roster</i> is survivorship-selected (today's liquid top-100), so roster membership is
 NOT fully point-in-time. Intra-window prices ARE point-in-time clean.</div>""")

    parts.append("""<div class=legend>
 <span style="background:#1b7d3e">edge &ge;+2pp &amp; null&ge;95</span>
 <span style="background:#2f9e54">+1..2pp &amp; null&ge;95</span>
 <span style="background:#3cb371">0..1pp &amp; null&ge;95</span>
 <span style="background:#6b8f3a">positive, weak null</span>
 <span style="background:#8a5a5a">mildly negative</span>
 <span style="background:#c0504d">-0.5..-2pp</span>
 <span style="background:#a52828">&le;-2pp</span>
 <span style="background:#3a3a3a;color:#888">abstain</span>
 </div>""")
    parts.append("<div class=meta>Cell = <b>OOS-2024 edge</b> (linear %/wk vs always-long). "
                 "Sub-lines: random-null percentile, deploy rate, OOS Sharpe; then VAL edge + OOS compound.</div>")

    for group, fams in FAMILY_GROUPS.items():
        parts.append(f"<h2>{html.escape(group)}</h2>")
        parts.append("<table><tr><th class=fam>Family</th>" +
                     "".join(f"<th>{tf}</th>" for tf in TFS) + "</tr>")
        for fam in fams:
            cells = "".join(cell_html(merged.get((fam, tf))) for tf in TFS)
            parts.append(f"<tr><td class=fam>{fam}</td>{cells}</tr>")
        parts.append("</table>")

    # quick ranked OOS leaderboard (consistent cells only)
    cons = []
    for (fam, tf), r in merged.items():
        v, o = r.get("VAL"), r.get("OOS2024")
        if v and o and v["edge_pp"] > 0 and o["edge_pp"] > 0 and o["deploy"] > 1e-3:
            cons.append((o["edge_pp"], fam, tf, r))
    cons.sort(reverse=True)
    parts.append("<h2>Top VAL/OOS-consistent cells (trustworthy — positive BOTH windows)</h2>")
    parts.append("<table><tr><th class=fam>Rank</th><th>Engine</th><th>TF</th><th>OOS edge/wk</th>"
                 "<th>OOS null</th><th>deploy</th><th>OOS Sharpe</th><th>profile</th></tr>")
    for i, (oe, fam, tf, r) in enumerate(cons[:20], 1):
        o = r["OOS2024"]
        parts.append(f"<tr><td class=fam>{i}</td><td class=fam>{fam}</td><td>{tf}</td>"
                     f"<td><b>{oe:+.2f}</b></td><td>{r['OOS_null_pct']:.0f}</td>"
                     f"<td>{o['deploy']:.0%}</td><td>{o['sharpe']:.2f}</td><td>{profile_tag(r)}</td></tr>")
    parts.append("</table>")

    parts.append(f"<div class=meta>{len(merged)} cells across {len(TFS)} timeframes. "
                 "1d/4h carry the most consistent signal; 2h/1h/30m are noisier (VAL&rarr;OOS sign flips). "
                 "Only ADX is genuinely selective (see DEADCASE_REWORK_RESULT.md).</div>")
    parts.append("</body></html>")
    return "\n".join(parts)


def main():
    args = [Path(a) for a in sys.argv[1:]]
    if not args:
        args = sorted(DNA_DIR.glob("ti_dna_*.json"), key=lambda p: p.stat().st_mtime)[-2:]
    asset_note = ("u100 roster = 77 assets with chimera data; 56 had &gt;20 bars in VAL, "
                  "63 in OOS-2024 (newer listings like EIGEN/ONDO/ZRO simply absent earlier).")
    out_html = build(args, asset_note)
    out = DNA_DIR / "DNA_REPORT.html"
    out.write_text(out_html, encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Merged {len(args)} JSON file(s): {[p.name for p in args]}")


if __name__ == "__main__":
    main()
