"""Look at built views for one asset and date range. Text summary, optional HTML chart."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from pipeline.constants import CORE_BAR_TYPES, TIME_WIDTH_MS
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
    for i, bt in enumerate(rows, start=1):
        df = views[bt]
        ts = [datetime.fromtimestamp(int(t) / 1000.0, tz=timezone.utc) for t in df["ts_ms"].to_list()]
        fig.add_trace(
            go.Candlestick(
                x=ts, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                name=bt, showlegend=False,
            ),
            row=i, col=1,
        )
        if i == 1 and "dollar" in views:
            d = views["dollar"]
            dts = [datetime.fromtimestamp(int(t) / 1000.0, tz=timezone.utc) for t in d["ts_ms"].to_list()]
            fig.add_trace(
                go.Scatter(
                    x=dts, y=d["close"], mode="markers",
                    marker=dict(size=5, color="#d62728"),
                    name="dollar close",
                ),
                row=1, col=1,
            )
    fig.update_layout(
        title=f"{asset}  {start} → {end}  (time bars share one axis; red markers are dollar-bar closes)",
        xaxis_rangeslider_visible=False,
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
