"""build_chart_gallery.py -- pictorial inventory of all data we have.

Generates a structured chart gallery for asset/cadence/chart-type combinations.
Output: runs/audit/CHART_GALLERY_<DATE>/<asset>/<cadence>/<chart>.png + INDEX.md.

Design:
  - 5 cadences (1d/4h/1h/15m/dollar) per asset
  - 6 sample assets covering DNA buckets:
      BTC (PRIME), ETH (PRIME), SOL (PRIME),
      SHIB (STEADY, recovered), AAVE (STEADY),
      WIF (DEGEN, recovered)
  - 12 chart types per asset/cadence combo where data permits:
      01 OHLC candles (last N bars)
      02 Volume profile + buy/sell
      03 Returns histogram
      04 norm_* feature time-series (small multiples)
      05 xd_* cross-asset overlay
      06 target_return distributions (h1/h4/h16/h64)
      07 Regime-colored price overlay
      08 Microstructure (kyle_lambda + vpin + hawkes_imbalance)
      09 Cumulative return + drawdown
      10 xrel_* features (cross-asset rank/ratio)
      11 RV / jump intensity overlay
      12 Funding + basis spread time-series
  - Cross-asset section: correlation heatmap, dispersion, leader/follower

Plus alternative bar-type comparisons:
  - Dollar vs DIB vs Range vs Runs(tick/vol) vs AdaptiveVol on BTC

Usage:
  python scripts/audit/build_chart_gallery.py
  python scripts/audit/build_chart_gallery.py --assets BTC ETH --cadences 1d 4h
  python scripts/audit/build_chart_gallery.py --dry-run

OUTPUT:
  runs/audit/CHART_GALLERY_2026_05_21/
    INDEX.md                    -- master gallery index with image links
    <ASSET>/
      <cadence>/
        01_candles.png, 02_volume.png, ...
    _cross_asset/
    _bar_types/
"""
from __future__ import annotations

__contract__ = {
    "kind": "diagnostic_script",
    "stage": "chart_gallery",
    "inputs": {"args": ["--assets", "--cadences", "--charts", "--dry-run", "--days"]},
    "outputs": {"dir": "runs/audit/CHART_GALLERY_<DATE>/", "INDEX": "INDEX.md"},
}

import argparse
import sys
import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# Configure matplotlib for headless Windows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 180   # bumped from 110 for clarity
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10

# 2026-05-21: dual-output -- save high-res PNG AND scalable SVG.
# Candles especially benefit from SVG (infinite zoom; no pixelation on detail).
SAVE_SVG = True
SAVE_BOTH = True

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
CHIMERA_DIR = ROOT / "data" / "processed" / "chimera"
BARS_DIR = ROOT / "data" / "processed" / "bars"

TODAY = datetime.now().strftime("%Y_%m_%d")
# 2026-05-21: relocated from runs/audit/ to plots/ for easier viewing.
OUT_DIR = ROOT / "plots" / f"charts_{TODAY}"

# Default sample — covers PRIME / STEADY / DEGEN buckets + recovered assets
DEFAULT_ASSETS = ["BTC", "ETH", "SOL", "SHIB", "AAVE", "WIF"]
DEFAULT_CADENCES = ["1d", "4h", "1h", "15m", "dollar"]

# Per-cadence default window for plotting (last N bars)
CADENCE_WINDOWS = {
    "1d":     {"bars": 365, "label": "365 days"},
    "4h":     {"bars": 600, "label": "~100 days"},
    "1h":     {"bars": 720, "label": "~30 days"},
    "15m":    {"bars": 960, "label": "~10 days"},
    "dollar": {"bars": 1500, "label": "1500 bars"},
}


def _load_chimera(asset_root: str, cadence: str, bars: int) -> pd.DataFrame | None:
    """Load the most recent chimera for asset/cadence, return last N bars."""
    sym = f"{asset_root.lower()}usdt"
    pattern = str(CHIMERA_DIR / cadence / f"{sym}_v51_chimera_{cadence}_*.parquet")
    files = sorted(glob.glob(pattern))
    if cadence == "dollar":
        # dollar files don't have cadence in name
        pattern = str(CHIMERA_DIR / "dollar" / f"{sym}_v51_chimera_*.parquet")
        files = sorted(glob.glob(pattern))
    if not files:
        return None
    f = files[-1]
    try:
        df = pl.scan_parquet(f).tail(bars).collect().to_pandas()
        df["dt"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        print(f"  [load_chimera] {asset_root} {cadence}: {type(e).__name__}: {e}")
        return None


def _save(fig, path: Path) -> None:
    """Save PNG (high-res) and SVG (vector). SVG enables infinite zoom for candles."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if SAVE_SVG:
        svg_path = path.with_suffix(".svg")
        try:
            fig.savefig(svg_path, bbox_inches="tight", format="svg")
        except Exception as e:
            print(f"    [warn] SVG save failed for {path.name}: {e}")
    plt.close(fig)


# ─── Chart functions ────────────────────────────────────────────────────────

def chart_01_candles(df: pd.DataFrame, asset: str, cadence: str, out: Path):
    """OHLC candlestick + volume (lower panel) — high-res via SVG + bigger figure."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10),
                                     gridspec_kw={"height_ratios": [3, 1]},
                                     sharex=True)
    # Use MEDIAN bar duration for width (robust against irregular cadences like dollar bars)
    if len(df) > 1:
        diffs = df["dt"].diff().dt.total_seconds().dropna()
        median_dur_s = diffs.median()
        # matplotlib expects width in days; ×0.7 leaves gap between bars
        bar_w = median_dur_s / 86400.0 * 0.7
    else:
        bar_w = 0.5
    up = df["close"] >= df["open"]
    dn = ~up

    # Wicks (thin vertical lines)
    ax1.vlines(df["dt"], df["low"], df["high"], color="#37474f", lw=0.6, alpha=0.85)
    # Bodies — distinct colors + opaque for clarity
    ax1.bar(df.loc[up, "dt"], (df.loc[up, "close"] - df.loc[up, "open"]),
            bottom=df.loc[up, "open"], color="#26a69a",
            width=bar_w, alpha=0.95, edgecolor="#004d40", linewidth=0.3)
    ax1.bar(df.loc[dn, "dt"], (df.loc[dn, "open"] - df.loc[dn, "close"]),
            bottom=df.loc[dn, "close"], color="#ef5350",
            width=bar_w, alpha=0.95, edgecolor="#b71c1c", linewidth=0.3)
    ax1.set_title(f"{asset} {cadence}  —  OHLC candles  ({CADENCE_WINDOWS[cadence]['label']}; "
                   f"{len(df):,} bars)", fontsize=12)
    ax1.set_ylabel("Price (USDT)", fontsize=10)

    # Volume — same width, semi-transparent
    ax2.bar(df.loc[up, "dt"], df.loc[up, "volume"], color="#26a69a", width=bar_w, alpha=0.7)
    ax2.bar(df.loc[dn, "dt"], df.loc[dn, "volume"], color="#ef5350", width=bar_w, alpha=0.7)
    ax2.set_ylabel("Volume", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.xaxis.set_major_locator(AutoDateLocator())
    ax2.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, out)


def chart_02_volume_breakdown(df: pd.DataFrame, asset: str, cadence: str, out: Path):
    """Buy/sell volume breakdown."""
    if "buy_vol" not in df.columns or "sell_vol" not in df.columns:
        return False
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax1.fill_between(df["dt"], df["buy_vol"], 0, color="#26a69a", alpha=0.7, label="buy_vol")
    ax1.fill_between(df["dt"], -df["sell_vol"], 0, color="#ef5350", alpha=0.7, label="sell_vol")
    ax1.set_title(f"{asset} {cadence}  -  Buy/Sell volume breakdown")
    ax1.set_ylabel("Volume (signed)")
    ax1.legend(loc="upper right", fontsize=8)
    # Imbalance ratio
    imb = (df["buy_vol"] - df["sell_vol"]) / (df["buy_vol"] + df["sell_vol"] + 1e-9)
    ax2.plot(df["dt"], imb, color="#1565c0", lw=0.8)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Imbalance (buy-sell)/(buy+sell)")
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    _save(fig, out)
    return True


def chart_03_returns_hist(df: pd.DataFrame, asset: str, cadence: str, out: Path):
    """Histogram of bar-to-bar returns + KDE + key percentiles."""
    rets = df["close"].pct_change().dropna()
    if len(rets) < 30:
        return False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(rets * 100, bins=80, color="#1565c0", alpha=0.7, edgecolor="black", lw=0.3)
    for p in [1, 5, 50, 95, 99]:
        q = np.percentile(rets * 100, p)
        ax1.axvline(q, color="red" if p in (1, 99) else "orange", lw=0.8, ls="--")
        ax1.text(q, ax1.get_ylim()[1] * (0.95 - 0.05 * [1, 5, 50, 95, 99].index(p)),
                  f"p{p}={q:+.2f}%", fontsize=7)
    ax1.set_title(f"{asset} {cadence}  -  Bar-to-bar return distribution (n={len(rets)})")
    ax1.set_xlabel("Return (%)")
    ax1.set_ylabel("Count")

    # Cumulative return
    cumret = (1 + rets).cumprod() - 1
    ax2.plot(df["dt"].iloc[1:], cumret * 100, color="#1565c0", lw=1.0)
    ax2.set_title(f"Cumulative return ({CADENCE_WINDOWS[cadence]['label']}): {cumret.iloc[-1]*100:+.1f}%")
    ax2.set_ylabel("Cumulative return (%)")
    ax2.set_xlabel("Date")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    _save(fig, out)
    return True


def chart_04_norm_features(df: pd.DataFrame, asset: str, cadence: str, out: Path):
    """Small multiples of key norm_* features."""
    candidates = ["norm_return_1", "norm_hl_spread", "norm_flow_imbalance",
                   "norm_hawkes_imbalance", "norm_kyle_lambda", "norm_vpin",
                   "norm_funding", "norm_whale", "norm_efficiency"]
    feats = [c for c in candidates if c in df.columns]
    if not feats:
        return False
    n = len(feats)
    rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(14, 2.5 * rows), sharex=True)
    axes = axes.flatten() if rows > 1 else [axes] if not isinstance(axes, np.ndarray) else axes
    for ax, feat in zip(axes, feats):
        ax.plot(df["dt"], df[feat], lw=0.6, color="#1565c0")
        ax.axhline(0, color="gray", lw=0.4, alpha=0.6)
        ax.set_title(feat, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[len(feats):]:
        ax.set_visible(False)
    fig.suptitle(f"{asset} {cadence}  -  norm_* features ({len(feats)} of 9 available)",
                  fontsize=11)
    fig.tight_layout()
    _save(fig, out)
    return True


def chart_05_xd_features(df: pd.DataFrame, asset: str, cadence: str, out: Path):
    """Cross-asset xd_* features over time."""
    xd_cols = sorted([c for c in df.columns if c.startswith("xd_")])
    if not xd_cols:
        return False
    n = len(xd_cols)
    rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(14, 2.5 * rows), sharex=True)
    axes = axes.flatten() if rows > 1 else [axes] if not isinstance(axes, np.ndarray) else axes
    for ax, feat in zip(axes, xd_cols):
        ax.plot(df["dt"], df[feat], lw=0.6, color="#6a1b9a")
        ax.axhline(0, color="gray", lw=0.4, alpha=0.6)
        ax.set_title(feat, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[len(xd_cols):]:
        ax.set_visible(False)
    fig.suptitle(f"{asset} {cadence}  -  Cross-asset xd_* features ({n} cols)",
                  fontsize=11)
    fig.tight_layout()
    _save(fig, out)
    return True


def chart_06_target_distributions(df: pd.DataFrame, asset: str, cadence: str, out: Path):
    """Target return distributions for h1/h4/h16/h64."""
    targets = [c for c in df.columns if c.startswith("target_return_")
                and not c.endswith("_raw")]
    if not targets:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, t in zip(axes, targets[:4]):
        data = df[t].dropna() * 100
        if len(data) == 0:
            ax.set_visible(False); continue
        ax.hist(data, bins=60, color="#1565c0", alpha=0.7, edgecolor="black", lw=0.3)
        ax.axvline(0, color="black", lw=0.5)
        ax.axvline(data.mean(), color="red", lw=0.7, ls="--",
                    label=f"mean={data.mean():+.3f}%")
        ax.axvline(np.percentile(data, 50), color="orange", lw=0.7, ls="--",
                    label=f"med={np.percentile(data,50):+.3f}%")
        ax.set_title(f"{t}  (n={len(data)})", fontsize=10)
        ax.set_xlabel("Return (%)")
        ax.legend(fontsize=7, loc="upper right")
    for ax in axes[len(targets):]:
        ax.set_visible(False)
    fig.suptitle(f"{asset} {cadence}  -  Forward-return target distributions", fontsize=11)
    fig.tight_layout()
    _save(fig, out)
    return True


def chart_07_regime_overlay(df: pd.DataFrame, asset: str, cadence: str, out: Path):
    """Price colored by regime_label."""
    if "regime_label" not in df.columns:
        return False
    fig, ax = plt.subplots(figsize=(14, 6))
    regimes = df["regime_label"].dropna().unique()
    colors = {"bull": "#26a69a", "chop": "#fbc02d", "bear": "#ff7043", "crash": "#c62828"}
    # Plot per-regime segments
    df_v = df.dropna(subset=["regime_label"]).copy()
    if len(df_v) == 0:
        return False
    df_v["regime_str"] = df_v["regime_label"].astype(str).str.lower()
    for reg, color in colors.items():
        mask = df_v["regime_str"] == reg
        if mask.any():
            ax.scatter(df_v.loc[mask, "dt"], df_v.loc[mask, "close"],
                        s=2, color=color, label=reg, alpha=0.7)
    # Overlay price line
    ax.plot(df["dt"], df["close"], color="black", lw=0.3, alpha=0.4)
    ax.set_title(f"{asset} {cadence}  -  Price colored by regime_label")
    ax.set_ylabel("Price (USDT)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    _save(fig, out)
    return True


def chart_08_microstructure(df: pd.DataFrame, asset: str, cadence: str, out: Path):
    """Kyle lambda, VPIN, Hawkes imbalance time series."""
    cols = ["norm_kyle_lambda", "norm_vpin", "norm_hawkes_imbalance",
            "norm_flow_imbalance"]
    avail = [c for c in cols if c in df.columns]
    if not avail:
        return False
    fig, axes = plt.subplots(len(avail), 1, figsize=(14, 2.5 * len(avail)), sharex=True)
    if len(avail) == 1:
        axes = [axes]
    for ax, feat in zip(axes, avail):
        ax.plot(df["dt"], df[feat], lw=0.6, color="#0d47a1")
        ax.axhline(0, color="gray", lw=0.4)
        ax.fill_between(df["dt"], df[feat], 0, alpha=0.2, color="#0d47a1")
        ax.set_title(feat, fontsize=9)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.suptitle(f"{asset} {cadence}  -  Microstructure features ({len(avail)} cols)",
                  fontsize=11)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, out)
    return True


def chart_09_cumret_dd(df: pd.DataFrame, asset: str, cadence: str, out: Path):
    """Cumulative return + drawdown subplot."""
    rets = df["close"].pct_change().fillna(0)
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum / peak - 1) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7),
                                     gridspec_kw={"height_ratios": [2, 1]},
                                     sharex=True)
    ax1.plot(df["dt"], (cum - 1) * 100, color="#1565c0", lw=1.0)
    ax1.fill_between(df["dt"], (cum - 1) * 100, 0,
                      where=(cum - 1) >= 0, color="#26a69a", alpha=0.3)
    ax1.fill_between(df["dt"], (cum - 1) * 100, 0,
                      where=(cum - 1) < 0, color="#ef5350", alpha=0.3)
    ax1.axhline(0, color="black", lw=0.5)
    ax1.set_title(f"{asset} {cadence}  -  Cumulative return + drawdown "
                   f"(final: {(cum.iloc[-1]-1)*100:+.1f}%, max DD: {dd.min():+.1f}%)")
    ax1.set_ylabel("Cum return (%)")

    ax2.fill_between(df["dt"], dd, 0, color="#c62828", alpha=0.6)
    ax2.plot(df["dt"], dd, color="#c62828", lw=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    _save(fig, out)
    return True


def chart_10_xrel_features(df: pd.DataFrame, asset: str, cadence: str, out: Path):
    """xrel_* cross-asset relative features."""
    xrel_cols = sorted([c for c in df.columns if c.startswith("xrel_")])
    if not xrel_cols:
        return False
    # Pick representative features per metric family
    plot_cols = []
    for prefix in ["xrel_rv_bpv_5m", "xrel_hbr_eta_total", "xrel_liq_long_usd",
                    "xrel_wh_whale_net_usd", "xrel_lob_kyle_lambda_mean",
                    "xrel_hbr_n_trades", "xrel_rv_rv_5m"]:
        for suffix in ["xrank", "xratio"]:
            col = f"{prefix}_{suffix}"
            if col in df.columns:
                plot_cols.append(col)
                break  # one per metric family
    if not plot_cols:
        plot_cols = xrel_cols[:6]
    n = len(plot_cols)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 2.5 * rows), sharex=True)
    axes = axes.flatten() if rows > 1 else [axes] if not isinstance(axes, np.ndarray) else axes
    for ax, feat in zip(axes, plot_cols):
        ax.plot(df["dt"], df[feat], lw=0.6, color="#00838f")
        ax.set_title(feat, fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[len(plot_cols):]:
        ax.set_visible(False)
    fig.suptitle(f"{asset} {cadence}  -  xrel_* cross-asset relatives ({n} samples; {len(xrel_cols)} total)",
                  fontsize=11)
    fig.tight_layout()
    _save(fig, out)
    return True


def chart_11_rv_jump(df: pd.DataFrame, asset: str, cadence: str, out: Path):
    """RV/BPV/jump intensity overlay (chimera has these for dollar+1d typically)."""
    candidates = ["rv_rv_5m", "rv_bpv_5m", "rv_jv_5m", "rv_jump_frac",
                   "rv_jump_intensity_30d"]
    avail = [c for c in candidates if c in df.columns]
    if not avail:
        return False
    fig, axes = plt.subplots(len(avail), 1, figsize=(14, 2 * len(avail)), sharex=True)
    if len(avail) == 1:
        axes = [axes]
    for ax, feat in zip(axes, avail):
        ax.plot(df["dt"], df[feat], lw=0.6, color="#5d4037")
        ax.fill_between(df["dt"], df[feat], 0, alpha=0.3, color="#5d4037")
        ax.set_title(feat, fontsize=9)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.suptitle(f"{asset} {cadence}  -  Realized vol + jumps", fontsize=11)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, out)
    return True


def chart_12_funding_basis(df: pd.DataFrame, asset: str, cadence: str, out: Path):
    """Funding rate + basis-related features."""
    cols = ["norm_funding", "norm_funding_momentum", "norm_oi_change",
            "norm_oi_price_divergence"]
    avail = [c for c in cols if c in df.columns]
    if not avail:
        return False
    fig, axes = plt.subplots(len(avail), 1, figsize=(14, 2 * len(avail)), sharex=True)
    if len(avail) == 1:
        axes = [axes]
    for ax, feat in zip(axes, avail):
        ax.plot(df["dt"], df[feat], lw=0.6, color="#4527a0")
        ax.axhline(0, color="gray", lw=0.4)
        ax.set_title(feat, fontsize=9)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.suptitle(f"{asset} {cadence}  -  Funding + OI features", fontsize=11)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, out)
    return True


# ─── Cross-asset section ────────────────────────────────────────────────────

def cross_asset_correlation_heatmap(assets: list[str], cadence: str, days: int,
                                      out: Path) -> bool:
    """Correlation matrix of asset close-to-close returns."""
    rets = {}
    for a in assets:
        df = _load_chimera(a, cadence, days)
        if df is None or "close" not in df.columns: continue
        # Dedupe timestamps (dollar bars can have duplicates across asset hot spots)
        df = df.drop_duplicates(subset="dt", keep="last")
        rets[a] = df.set_index("dt")["close"].pct_change()
    if len(rets) < 2:
        return False
    panel = pd.DataFrame(rets).dropna(how="all")
    corr = panel.corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                     fontsize=7, color="black")
    ax.set_title(f"Cross-asset return correlation  ({cadence}, last {days} bars)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, out)
    return True


def cross_asset_normalized_price(assets: list[str], cadence: str, days: int,
                                    out: Path) -> bool:
    """Normalized cumulative-return overlay across assets."""
    fig, ax = plt.subplots(figsize=(14, 7))
    for a in assets:
        df = _load_chimera(a, cadence, days)
        if df is None: continue
        df = df.drop_duplicates(subset="dt", keep="last")
        norm = df["close"] / df["close"].iloc[0]
        ax.plot(df["dt"], norm, lw=1.0, label=a, alpha=0.85)
    ax.axhline(1.0, color="black", lw=0.5)
    ax.set_title(f"Normalized price overlay  ({cadence}, last {days} bars; price/price[0])")
    ax.set_ylabel("Normalized price")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9, loc="upper left")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    _save(fig, out)
    return True


# ─── Bar-type comparison ────────────────────────────────────────────────────

def chart_bar_type_overlay(asset: str, out: Path) -> bool:
    """Compare different bar types for the same asset over a recent window."""
    sym_u = asset.upper() + "USDT"
    sources = [
        ("dollar", CHIMERA_DIR / "dollar", f"{sym_u.lower()}_v51_chimera_*.parquet", "close"),
        ("dib",    BARS_DIR / "dib",       f"{sym_u}_dib_*.parquet",                 "close"),
        ("range",  BARS_DIR / "range",     f"{sym_u}_range_*.parquet",               "close"),
        ("adaptive_vol", BARS_DIR / "adaptive_vol", f"{sym_u}_adaptive_vol_*.parquet", "close"),
    ]
    fig, ax = plt.subplots(figsize=(14, 7))
    any_plotted = False
    for name, dir_, pattern, col in sources:
        files = sorted(dir_.glob(pattern))
        if not files: continue
        try:
            df = pl.scan_parquet(files[-1]).tail(2000).collect().to_pandas()
        except Exception:
            continue
        if "timestamp" not in df.columns or col not in df.columns:
            continue
        df["dt"] = pd.to_datetime(df["timestamp"], unit="ms")
        ax.plot(df["dt"], df[col], lw=0.8, label=f"{name} ({len(df)} bars)", alpha=0.75)
        any_plotted = True
    if not any_plotted:
        return False
    ax.set_title(f"{asset}  -  Bar-type comparison (last 2,000 bars per type)")
    ax.set_ylabel("Price (USDT)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9, loc="upper left")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    _save(fig, out)
    return True


# ─── Driver ─────────────────────────────────────────────────────────────────

CHART_FUNCS = [
    ("01_candles",          chart_01_candles),
    ("02_volume_breakdown", chart_02_volume_breakdown),
    ("03_returns_hist",     chart_03_returns_hist),
    ("04_norm_features",    chart_04_norm_features),
    ("05_xd_features",      chart_05_xd_features),
    ("06_target_distributions", chart_06_target_distributions),
    ("07_regime_overlay",   chart_07_regime_overlay),
    ("08_microstructure",   chart_08_microstructure),
    ("09_cumret_dd",        chart_09_cumret_dd),
    ("10_xrel_features",    chart_10_xrel_features),
    ("11_rv_jump",          chart_11_rv_jump),
    ("12_funding_basis",    chart_12_funding_basis),
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS,
                    help=f"Assets to plot (default: {DEFAULT_ASSETS}).")
    ap.add_argument("--cadences", nargs="+", default=DEFAULT_CADENCES,
                    choices=["1d", "4h", "1h", "15m", "dollar"],
                    help="Cadences to plot.")
    ap.add_argument("--charts", nargs="+", default=None,
                    help=f"Subset of chart names (default: all 12).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan only, no plots written.")
    return ap.parse_args()


def main():
    args = parse_args()
    print(f"[gallery] output dir: {OUT_DIR.relative_to(ROOT)}")
    print(f"[gallery] assets: {args.assets}")
    print(f"[gallery] cadences: {args.cadences}")
    chart_filter = set(args.charts) if args.charts else None
    funcs = [(n, f) for n, f in CHART_FUNCS if (chart_filter is None or n in chart_filter)]
    print(f"[gallery] charts: {len(funcs)} types per (asset, cadence)")

    if args.dry_run:
        n_plots = len(args.assets) * len(args.cadences) * len(funcs)
        print(f"[gallery] DRY-RUN: would generate up to {n_plots} per-asset plots "
              f"+ {len(args.cadences)*2} cross-asset + {len(args.assets)} bar-type")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory: list[dict] = []
    t0 = pd.Timestamp.utcnow()

    # Per-asset/cadence/chart-type loop
    for asset in args.assets:
        for cadence in args.cadences:
            bars = CADENCE_WINDOWS[cadence]["bars"]
            df = _load_chimera(asset, cadence, bars)
            if df is None or len(df) < 30:
                print(f"  [skip] {asset} {cadence}: no data or <30 bars")
                continue
            print(f"  [{asset} {cadence}] {len(df)} bars, {len(df.columns)} cols")
            cdir = OUT_DIR / asset / cadence
            cdir.mkdir(parents=True, exist_ok=True)
            for name, fn in funcs:
                path = cdir / f"{name}.png"
                try:
                    result = fn(df, asset, cadence, path)
                    ok = result is not False
                    inventory.append({"asset": asset, "cadence": cadence,
                                       "chart": name, "ok": ok,
                                       "path": str(path.relative_to(OUT_DIR))})
                except Exception as e:
                    print(f"    [FAIL] {name}: {type(e).__name__}: {e}")
                    inventory.append({"asset": asset, "cadence": cadence,
                                       "chart": name, "ok": False,
                                       "path": str(path.relative_to(OUT_DIR)),
                                       "error": f"{type(e).__name__}: {e}"})

    # Cross-asset section
    print(f"\n  [cross_asset]")
    xa_dir = OUT_DIR / "_cross_asset"
    xa_dir.mkdir(parents=True, exist_ok=True)
    for cadence in args.cadences:
        bars = CADENCE_WINDOWS[cadence]["bars"]
        try:
            ok1 = cross_asset_correlation_heatmap(args.assets, cadence, bars,
                                                    xa_dir / f"corr_{cadence}.png")
            ok2 = cross_asset_normalized_price(args.assets, cadence, bars,
                                                  xa_dir / f"norm_price_{cadence}.png")
            inventory.append({"asset": "_cross_asset", "cadence": cadence,
                               "chart": "correlation", "ok": ok1,
                               "path": f"_cross_asset/corr_{cadence}.png"})
            inventory.append({"asset": "_cross_asset", "cadence": cadence,
                               "chart": "norm_price", "ok": ok2,
                               "path": f"_cross_asset/norm_price_{cadence}.png"})
        except Exception as e:
            print(f"    [cross_asset {cadence}] FAIL: {type(e).__name__}: {e}")

    # Bar-type comparison
    print(f"\n  [bar_types]")
    bt_dir = OUT_DIR / "_bar_types"
    bt_dir.mkdir(parents=True, exist_ok=True)
    for asset in args.assets:
        try:
            ok = chart_bar_type_overlay(asset, bt_dir / f"{asset}_bar_types.png")
            inventory.append({"asset": "_bar_types", "cadence": "mixed",
                               "chart": asset, "ok": ok,
                               "path": f"_bar_types/{asset}_bar_types.png"})
        except Exception as e:
            print(f"    [{asset} bar_types] FAIL: {type(e).__name__}: {e}")

    elapsed = (pd.Timestamp.utcnow() - t0).total_seconds()

    # ─── Build INDEX.md ─────────────────────────────────────────────────
    n_ok = sum(1 for r in inventory if r["ok"])
    n_fail = sum(1 for r in inventory if not r["ok"])
    lines = [
        f"# Chart Gallery — {TODAY.replace('_','-')}\n",
        f"**Assets**: {', '.join(args.assets)}",
        f"**Cadences**: {', '.join(args.cadences)}",
        f"**Charts**: {len(funcs)} per (asset, cadence) + cross-asset + bar-types",
        f"**Generated**: {n_ok} plots ({n_fail} skipped); elapsed {elapsed:.0f}s",
        "",
        "## Cross-asset views",
        "",
    ]
    for cad in args.cadences:
        lines.extend([
            f"### {cad}",
            f"- ![corr](_cross_asset/corr_{cad}.png) — return correlation",
            f"- ![norm](_cross_asset/norm_price_{cad}.png) — normalized price overlay",
            ""
        ])
    lines.extend(["## Bar-type comparison", ""])
    for asset in args.assets:
        lines.append(f"- ![{asset}](_bar_types/{asset}_bar_types.png) — {asset} dollar vs DIB vs range vs adaptive_vol")
    lines.append("")
    lines.append("## Per-asset, per-cadence")
    lines.append("")
    for asset in args.assets:
        lines.append(f"### {asset}\n")
        for cadence in args.cadences:
            adir = OUT_DIR / asset / cadence
            if not adir.exists():
                continue
            lines.append(f"#### {asset} {cadence}\n")
            for name, _ in funcs:
                path = adir / f"{name}.png"
                svg_path = adir / f"{name}.svg"
                if path.exists():
                    rel = path.relative_to(OUT_DIR).as_posix()
                    svg_rel = svg_path.relative_to(OUT_DIR).as_posix() if svg_path.exists() else None
                    if svg_rel:
                        lines.append(f"- ![{name}]({rel}) — {name} ([SVG zoom-friendly]({svg_rel}))")
                    else:
                        lines.append(f"- ![{name}]({rel}) — {name}")
            lines.append("")
        lines.append("")

    (OUT_DIR / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] Gallery built: {n_ok} plots in {elapsed:.0f}s")
    print(f"     INDEX: {OUT_DIR.relative_to(ROOT)}/INDEX.md")

    # Also write a CSV inventory
    pd.DataFrame(inventory).to_csv(OUT_DIR / "inventory.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
