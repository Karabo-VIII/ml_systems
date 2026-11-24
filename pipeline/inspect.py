"""Look at built views for one asset and date range. Text summary, optional HTML chart."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from pipeline.constants import CORE_BAR_TYPES, DAY_MS, HOUR_MS, TIME_WIDTH_MS
from pipeline.partitions import load_partitions
from pipeline.paths import default_data_root
from pipeline.symbols import canonical_instrument
from pipeline.validate_run import clip_frame, range_ms

KEY_COLS = (
    "close", "volume", "volume_usd", "n_trades",
    "funding_rate", "norm_funding", "stbl_supply_usd", "soc_wiki_views",
    "etf_total_usdm",
)


def _utc(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_view(root: Path, layer: str, bar_type: str, asset: str, start: date, end: date) -> pl.DataFrame | None:
    if layer == "gold":
        directory = root / "gold" / bar_type
    else:
        directory = root / "silver" / "bars" / bar_type
    if not directory.exists():
        return None
    df = load_partitions(directory, asset)
    if df is None or df.is_empty():
        return None
    return clip_frame(df, start, end)


def collect_views(
    root: Path,
    asset: str,
    start: date,
    end: date,
    bar_types: list[str] | None = None,
) -> dict[str, pl.DataFrame]:
    types = list(bar_types or CORE_BAR_TYPES)
    out: dict[str, pl.DataFrame] = {}
    for bt in types:
        # Prefer gold (features + bars) when present; fall back to silver bars.
        df = _load_view(root, "gold", bt, asset, start, end)
        if df is None or df.is_empty():
            df = _load_view(root, "silver", bt, asset, start, end)
        if df is not None and not df.is_empty():
            out[bt] = df.sort("ts_ms")
    return out


def _null_summary(df: pl.DataFrame) -> list[str]:
    parts = []
    seen = set()
    for col in KEY_COLS:
        if col not in df.columns:
            continue
        seen.add(col)
        parts.append(f"{col}={int(df[col].null_count())}/{df.height}")
    # Mention other all-null feature-like columns briefly.
    extra = []
    for col in df.columns:
        if col in seen or col in {"asset", "bar_type", "bar_id", "split"}:
            continue
        if df[col].null_count() == df.height and df.height:
            extra.append(col)
    if extra:
        parts.append("all-null=" + ",".join(extra[:8]) + ("..." if len(extra) > 8 else ""))
    return parts


def _rel(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-8 + 1e-9 * max(abs(a), abs(b))


def format_inspect(
    asset: str,
    start: date,
    end: date,
    root: Path,
    views: dict[str, pl.DataFrame],
) -> str:
    lines = [f"inspect  asset={asset}  {start} -> {end}  root={root}"]
    if not views:
        lines.append("  no gold or silver bars found for this asset and range")
        return "\n".join(lines)
    for bt, df in views.items():
        first = _utc(int(df["ts_ms"].min()))
        last = _utc(int(df["ts_ms"].max()))
        lines.append(f"  {bt:<12} rows={df.height:<7} first={first}  last={last}")
        nulls = _null_summary(df)
        if nulls:
            lines.append(f"               nulls: {', '.join(nulls)}")
    one_m = views.get("time_1m")
    one_d = views.get("time_1d")
    if one_m is not None and one_d is not None and "volume" in one_m.columns:
        v1 = float(one_m["volume"].sum())
        vd = float(one_d["volume"].sum())
        match = "yes" if _rel(v1, vd) else "NO"
        lines.append(f"  reconcile    sum(time_1m.volume)={v1:.8g}  sum(time_1d.volume)={vd:.8g}  match={match}")
        closes = []
        for name in ("time_1m", "time_1h", "time_1d", "dollar"):
            if name in views and "close" in views[name].columns:
                closes.append(f"{name}={float(views[name]['close'][-1]):.6g}")
        if closes:
            # 1m last close should agree with 1d last close (same last trade of the last day).
            agree = ""
            if one_m is not None and one_d is not None:
                c1 = float(one_m["close"][-1])
                c2 = float(one_d["close"][-1])
                agree = "  1m_vs_1d=" + ("yes" if _rel(c1, c2) else "NO")
            lines.append("  last close   " + "  ".join(closes) + agree)
    return "\n".join(lines)



def candle_x(ts_open_ms: int, ts_ms: int) -> tuple[int, int]:
    """Return ``(mid_ms, width_ms)`` so a candle covers ``(ts_open_ms, ts_ms]``.

    Bars are labelled at ``ts_ms = t_close``. Plot at the midpoint of the
    interval with width equal to the bar duration.
    """
    open_ms = int(ts_open_ms)
    close_ms = int(ts_ms)
    width = close_ms - open_ms
    mid = open_ms + width // 2
    return mid, width


def _open_ms_list(df: pl.DataFrame, bar_type: str) -> list[int]:
    closes = [int(t) for t in df["ts_ms"].to_list()]
    if "ts_open_ms" in df.columns:
        opens = df["ts_open_ms"].to_list()
        out = []
        for open_ms, close_ms in zip(opens, closes):
            if open_ms is None:
                width = TIME_WIDTH_MS.get(bar_type)
                out.append(close_ms - width if width is not None else close_ms)
            else:
                out.append(int(open_ms))
        return out
    width = TIME_WIDTH_MS.get(bar_type)
    if width is None:
        raise ValueError(f"cannot infer ts_open_ms for {bar_type}")
    return [c - width for c in closes]


def write_html_chart(
    path: Path,
    asset: str,
    start: date,
    end: date,
    views: dict[str, pl.DataFrame],
) -> Path:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover - exercised when viz extra is missing
        raise SystemExit(
            "HTML charts need the viz extra: pip install -e \".[viz]\""
        ) from exc

    rows = [bt for bt in ("time_1m", "time_1h", "time_1d") if bt in views]
    if not rows:
        raise SystemExit("no time-bar views to chart")
    fig = make_subplots(
        rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.04,
        subplot_titles=tuple(f"{asset} {bt}" for bt in rows),
    )
    lo, hi = range_ms(start, end)
    up, down = "#26a69a", "#ef5350"
    for i, bt in enumerate(rows, start=1):
        df = views[bt]
        opens = _open_ms_list(df, bt)
        closes = [int(t) for t in df["ts_ms"].to_list()]
        mids = []
        widths = []
        hover = []
        for open_ms, close_ms in zip(opens, closes):
            mid, width = candle_x(open_ms, close_ms)
            mids.append(datetime.fromtimestamp(mid / 1000.0, tz=timezone.utc))
            widths.append(width)
            hover.append([_utc(open_ms), _utc(close_ms)])
        o = df["open"].to_list()
        h = df["high"].to_list()
        low = df["low"].to_list()
        c = df["close"].to_list()
        colors = [up if (cv or 0) >= (ov or 0) else down for ov, cv in zip(o, c)]
        wick_x: list[datetime | None] = []
        wick_y: list[float | None] = []
        for mid, lo_px, hi_px in zip(mids, low, h):
            wick_x.extend([mid, mid, None])
            wick_y.extend([lo_px, hi_px, None])
        fig.add_trace(
            go.Scatter(
                x=wick_x, y=wick_y, mode="lines",
                line=dict(width=1, color="#444444"),
                hoverinfo="skip", showlegend=False, name=f"{bt} wick",
            ),
            row=i, col=1,
        )
        body = []
        base = []
        for ov, cv, lo_px, hi_px in zip(o, c, low, h):
            ov = 0.0 if ov is None else float(ov)
            cv = 0.0 if cv is None else float(cv)
            top, bot = (cv, ov) if cv >= ov else (ov, cv)
            height = top - bot
            if height <= 0:
                span = (hi_px or top) - (lo_px or bot)
                height = span * 0.002 if span else 1e-9
                bot = (ov + cv) / 2.0 - height / 2.0
            body.append(height)
            base.append(bot)
        fig.add_trace(
            go.Bar(
                x=mids, y=body, base=base, width=widths,
                marker_color=colors, marker_line_width=0,
                name=bt, showlegend=False,
                customdata=[[ho[0], ho[1], ov, hi, lo_v, cv] for ho, ov, hi, lo_v, cv in zip(hover, o, h, low, c)],
                hovertemplate=(
                    "open time %{customdata[0]}<br>"
                    "close time %{customdata[1]}<br>"
                    "open %{customdata[2]}<br>"
                    "high %{customdata[3]}<br>"
                    "low %{customdata[4]}<br>"
                    "close %{customdata[5]}"
                    "<extra></extra>"
                ),
            ),
            row=i, col=1,
        )
        if i == 1 and "dollar" in views:
            d = views["dollar"]
            d_open = _open_ms_list(d, "dollar") if "ts_open_ms" in d.columns else [int(t) for t in d["ts_ms"].to_list()]
            d_close = [int(t) for t in d["ts_ms"].to_list()]
            dts = [datetime.fromtimestamp(t / 1000.0, tz=timezone.utc) for t in d_close]
            d_hover = [[_utc(o_ms), _utc(c_ms)] for o_ms, c_ms in zip(d_open, d_close)]
            fig.add_trace(
                go.Scatter(
                    x=dts, y=d["close"], mode="markers",
                    marker=dict(size=5, color="#d62728"),
                    name="dollar close",
                    customdata=d_hover,
                    hovertemplate=(
                        "dollar close %{y}<br>"
                        "open time %{customdata[0]}<br>"
                        "close time %{customdata[1]}"
                        "<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )
    fig.update_layout(
        title=f"{asset}  {start} → {end}  (candles span [open, close]; red markers are dollar-bar closes)",
        xaxis_rangeslider_visible=False,
        barmode="overlay",
        bargap=0,
        height=320 * len(rows) + 80,
        margin=dict(l=60, r=20, t=80, b=40),
        legend=dict(orientation="h", y=1.04),
    )
    for i in range(1, len(rows) + 1):
        fig.update_xaxes(rangeslider_visible=False, row=i, col=1)
        if lo is not None and hi is not None:
            fig.update_xaxes(
                range=[
                    datetime.fromtimestamp(lo / 1000.0, tz=timezone.utc),
                    datetime.fromtimestamp(hi / 1000.0, tz=timezone.utc),
                ],
                row=i, col=1,
            )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return path


# ---------------------------------------------------------------- QA report (every view)

# Bar columns and metadata; everything else numeric in gold is a feature or label.
NON_FEATURE = {
    "asset", "bar_type", "bar_id", "ts_open_ms", "ts_ms", "open", "high", "low", "close", "vwap",
    "volume", "volume_usd", "buy_volume", "sell_volume", "buy_usd", "sell_usd", "n_trades",
    "signed_usd", "split", "is_u10", "is_u50", "is_u100", "asset_dna", "listing_date", "is_warmup",
}
LINE_MAX = 20_000
ENVELOPE_VIEWS = ("time_1m", "time_5m")  # too many bars for a line: plotted as an hourly high/low band
EXTERNAL_COLS = ("stbl_supply_usd", "soc_wiki_views", "dv_btc_dvol_close", "funding_day_mean")


def view_stats(root: Path, bar_type: str, asset: str, start: date, end: date) -> dict | None:
    """Everything the report draws for one view, reduced to small frames."""
    df = _load_view(root, "gold", bar_type, asset, start, end)
    if df is None or df.is_empty():
        return None
    df = df.sort("ts_ms")
    feats = [c for c in df.columns if c not in NON_FEATURE and df[c].dtype.is_numeric()]
    settled = df.filter(~pl.col("is_warmup").fill_null(True)) if "is_warmup" in df.columns else df
    coverage, scale = {}, {}
    for c in feats:
        s = settled[c].cast(pl.Float64)
        coverage[c] = 1.0 - (s.null_count() + s.is_nan().sum()) / settled.height if settled.height else 0.0
        finite = s.filter(s.is_finite())
        scale[c] = float(finite.abs().quantile(0.99)) if finite.len() else None
    daily = (
        df.group_by((pl.col("ts_open_ms") // DAY_MS).alias("day"))
        .agg(pl.col("volume_usd").sum(), pl.len().alias("bars"))
        .sort("day")
    )
    if bar_type in ENVELOPE_VIEWS:
        price = (
            df.group_by((pl.col("ts_open_ms") // HOUR_MS * HOUR_MS + HOUR_MS).alias("ts_ms"))
            .agg(pl.col("high").max(), pl.col("low").min(), pl.col("close").last())
            .sort("ts_ms")
        )
    else:
        step = max(1, df.height // LINE_MAX)
        price = df.select(["ts_ms", "close"]).gather_every(step)
    extra = [c for c in ("funding_rate", *EXTERNAL_COLS) if c in df.columns]
    return {
        "rows": df.height,
        "first": _utc(int(df["ts_ms"].min())),
        "last": _utc(int(df["ts_ms"].max())),
        "coverage": coverage,
        "scale": scale,
        "daily": daily,
        "price": price,
        "series": df.select(["ts_ms", *extra]) if bar_type in ("time_8h", "time_1d") else None,
    }


def _severity_color(sev: str) -> str:
    return {"PASS": "#1b7f3b", "WARN": "#b26a00", "FAIL": "#b3261e"}.get(sev, "#666666")


def _crosscheck_rows(root: Path, asset: str) -> list[tuple[str, str, str]]:
    path = root / "crosscheck" / "report.json"
    if not path.exists():
        return []
    import json
    data = json.loads(path.read_text())
    return [(c["name"], c["severity"], c["detail"]) for c in data.get("checks", [])
            if c["detail"].startswith(asset)]


def _html_table(rows: list[tuple[str, str, str]]) -> str:
    import html as _h
    body = "".join(
        f"<tr><td>{_h.escape(n)}</td><td style='color:{_severity_color(s)};font-weight:600'>{s}</td>"
        f"<td>{_h.escape(d)}</td></tr>" for n, s, d in rows
    )
    return f"<table class='t'><tr><th>check</th><th>result</th><th>detail</th></tr>{body}</table>"


PAGE_CSS = """<style>
body{font-family:system-ui,sans-serif;margin:16px;background:#fff;color:#111}
h1{font-size:20px} h2{font-size:16px;margin-top:20px}
.t{border-collapse:collapse;font-size:12px;width:100%} .t td,.t th{border:1px solid #ddd;padding:3px 6px;text-align:left;vertical-align:top}
.t th{background:#f4f4f4}
</style>"""


def write_asset_report(path: Path, root: Path, asset: str, start: date, end: date,
                       bar_types: list[str] | None = None) -> dict:
    """One page per asset: every view on shared axes, reconciliations, feature coverage and scale."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    types = list(bar_types or CORE_BAR_TYPES)
    stats = {bt: s for bt in types if (s := view_stats(root, bt, asset, start, end)) is not None}
    if not stats:
        raise SystemExit(f"no gold views for {asset}")
    base = stats.get("time_1m")
    recon: dict[str, float] = {}
    if base is not None:
        b = base["daily"].rename({"volume_usd": "base"})
        for bt, s in stats.items():
            j = s["daily"].join(b, on="day", how="full", coalesce=True)
            rel = ((j["volume_usd"].fill_null(0) - j["base"].fill_null(0)).abs() / j["base"].fill_null(0).clip(1e-9))
            recon[bt] = float(rel.max())
    titles = (
        "close in every view (1m/5m drawn as an hourly high-low band; dots: dollar bars)",
        "notional per UTC day, every view (lines should coincide; dollar bars are bucketed by the day they open)",
        "bars per day (1m: 1440 when no minute is missing; dollar: event count)",
        "funding_rate (8h gold) and the daily external series (1d gold, each scaled to its max)",
        "feature coverage after warmup (share non-null), view x column",
        "feature scale: log10 of p99 |value| after warmup, view x column",
    )
    fig = make_subplots(rows=6, cols=1, subplot_titles=titles, vertical_spacing=0.045,
                        row_heights=[0.2, 0.13, 0.09, 0.12, 0.23, 0.23])
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#d62728"]
    color = {bt: palette[i % len(palette)] for i, bt in enumerate(types)}
    for bt, s in stats.items():
        p = s["price"]
        x = [_dt(t) for t in p["ts_ms"].to_list()]
        if bt in ENVELOPE_VIEWS:
            fig.add_trace(go.Scattergl(x=x, y=p["high"], mode="lines", line=dict(width=0.6, color=color[bt]),
                                       name=f"{bt} hourly high", legendgroup=bt), row=1, col=1)
            fig.add_trace(go.Scattergl(x=x, y=p["low"], mode="lines", line=dict(width=0.6, color=color[bt]),
                                       name=f"{bt} hourly low", legendgroup=bt, showlegend=False), row=1, col=1)
        elif bt == "dollar":
            fig.add_trace(go.Scattergl(x=x, y=p["close"], mode="markers", marker=dict(size=2, color=color[bt]),
                                       name="dollar close"), row=1, col=1)
        else:
            fig.add_trace(go.Scattergl(x=x, y=p["close"], mode="lines", line=dict(width=1, color=color[bt]),
                                       name=f"{bt} close"), row=1, col=1)
        d = s["daily"]
        dx = [_dt(int(t) * DAY_MS) for t in d["day"].to_list()]
        fig.add_trace(go.Scattergl(x=dx, y=d["volume_usd"], mode="lines", line=dict(width=1, color=color[bt]),
                                   name=f"{bt} notional/day", legendgroup=bt, showlegend=False), row=2, col=1)
        if bt in ("time_1m", "dollar"):
            fig.add_trace(go.Scattergl(x=dx, y=d["bars"], mode="lines", line=dict(width=1, color=color[bt]),
                                       name=f"{bt} bars/day", legendgroup=bt, showlegend=False), row=3, col=1)
    for bt, cols in (("time_8h", ("funding_rate",)), ("time_1d", EXTERNAL_COLS)):
        s = stats.get(bt)
        if s is None or s["series"] is None:
            continue
        ser = s["series"]
        x = [_dt(t) for t in ser["ts_ms"].to_list()]
        for c in cols:
            if c not in ser.columns or ser[c].null_count() == ser.height:
                continue
            y = ser[c]
            top = float(y.abs().max() or 1.0)
            fig.add_trace(go.Scattergl(x=x, y=y / top, mode="lines",
                                       line=dict(width=1), name=f"{c} (max {top:.3g})"), row=4, col=1)
    cols = sorted({c for s in stats.values() for c in s["coverage"]})
    views = list(stats)
    cov = [[stats[v]["coverage"].get(c) for v in views] for c in cols]
    sc = [[(np.log10(x) if x and x > 0 else None) for x in (stats[v]["scale"].get(c) for v in views)] for c in cols]
    fig.add_trace(go.Heatmap(z=cov, x=views, y=cols, zmin=0, zmax=1, colorscale="Viridis", showscale=False,
                             hovertemplate="%{y} @ %{x}: %{z:.3f}<extra></extra>"), row=5, col=1)
    fig.add_trace(go.Heatmap(z=sc, x=views, y=cols, colorscale="RdBu_r", zmid=0, showscale=False,
                             hovertemplate="%{y} @ %{x}: log10 p99|v| = %{z:.2f}<extra></extra>"), row=6, col=1)
    for r in (2, 3, 4):
        fig.update_xaxes(matches="x", row=r, col=1)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=2, col=1)
    fig.update_yaxes(tickfont=dict(size=7), row=5, col=1)
    fig.update_yaxes(tickfont=dict(size=7), row=6, col=1)
    fig.update_layout(height=3400, margin=dict(l=170, r=20, t=90, b=30),
                      legend=dict(orientation="h", yanchor="bottom", y=1.012, font=dict(size=10)))
    import html as _h
    view_rows = "".join(
        f"<tr><td>{bt}</td><td>{s['rows']}</td><td>{s['first']}</td><td>{s['last']}</td>"
        f"<td>{recon.get(bt, float('nan')):.2e}</td></tr>" for bt, s in stats.items()
    )
    head = (
        f"<h1>{_h.escape(asset)} &middot; {start} to {end}</h1>"
        "<h2>Views</h2><table class='t'><tr><th>view</th><th>rows</th><th>first close (UTC)</th>"
        "<th>last close (UTC)</th><th>max daily notional diff vs 1m (relative)</th></tr>"
        f"{view_rows}</table>"
    )
    cc = _crosscheck_rows(root, asset)
    if cc:
        head += "<h2>Public-API cross-check</h2>" + _html_table(cc)
    page = fig.to_html(full_html=True, include_plotlyjs="cdn")
    page = page.replace("<head>", "<head><meta charset='utf-8'><title>" + _h.escape(asset) + " views</title>" + PAGE_CSS, 1)
    page = page.replace("<body>", "<body>" + head, 1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return {"views": {bt: s["rows"] for bt, s in stats.items()}, "recon": recon}


def write_report(out_dir: Path, root: Path, assets: list[str], start: date, end: date,
                 bar_types: list[str] | None = None) -> Path:
    """An index page plus one page per asset (see ``write_asset_report``)."""
    import html as _h
    out_dir = Path(out_dir)
    rows = []
    for asset in assets:
        print(f"[report] {asset}", flush=True)
        info = write_asset_report(out_dir / f"{asset}.html", root, asset, start, end, bar_types)
        worst_recon = max((v for k, v in info["recon"].items()), default=float("nan"))
        cc = _crosscheck_rows(root, asset)
        sev = {"FAIL": 0, "WARN": 0, "PASS": 0}
        for _, s, _d in cc:
            sev[s] = sev.get(s, 0) + 1
        rows.append(
            f"<tr><td><a href='{_h.escape(asset)}.html'>{_h.escape(asset)}</a></td>"
            f"<td>{len(info['views'])}</td><td>{sum(info['views'].values())}</td><td>{worst_recon:.2e}</td>"
            f"<td>{sev['PASS']}</td><td style='color:{_severity_color('WARN')}'>{sev['WARN']}</td>"
            f"<td style='color:{_severity_color('FAIL')}'>{sev['FAIL']}</td></tr>"
        )
    page = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Views report</title>" + PAGE_CSS + "</head><body>"
        f"<h1>Views report &middot; {start} to {end} &middot; {_h.escape(str(root))}</h1>"
        "<table class='t'><tr><th>asset</th><th>views</th><th>bars</th><th>worst daily notional diff vs 1m</th>"
        "<th>cross-check PASS</th><th>WARN</th><th>FAIL</th></tr>" + "".join(rows) + "</table></body></html>"
    )
    index = out_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


def _dt(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)


def inspect(
    *,
    asset: str,
    start: date,
    end: date,
    data_root: Path | None = None,
    html: Path | None = None,
    bar_types: list[str] | None = None,
) -> int:
    root = Path(data_root) if data_root is not None else default_data_root()
    if not root.is_absolute():
        from pipeline.paths import repo_root
        root = (repo_root() / root).resolve()
    asset = canonical_instrument(asset)
    views = collect_views(root, asset, start, end, bar_types)
    print(format_inspect(asset, start, end, root, views))
    if html is not None:
        written = write_html_chart(html, asset, start, end, views)
        print(f"  html         {written}")
    return 0 if views else 2
