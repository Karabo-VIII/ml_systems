"""src/narrate/state.py -- turn a chimera window into structured FAMILY READS.

Given a chimera dataframe, a PERIOD mask (the window to narrate), and a REFERENCE mask (the history to compare
against, normally the asset's full available history), this computes for every feature family:
  - a DIRECTION (bullish / bearish / neutral) from polarity-weighted feature values,
  - an INTENSITY percentile (where this window sits vs the asset's own history),
  - the SALIENT features driving the read, and
  - notable EVENTS inside the window (liquidation cascades, basis panic/frenzy, funding flips, vol expansions).

All reads are PER-WINDOW (a multi-candle MODE), never per-bar. Percentile-vs-own-history makes every number
human-interpretable ("momentum sits at the 82nd percentile of this asset's history").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .feature_map import FAMILIES, FAMILY_ORDER, FEATURES, classify, group_columns


@dataclass
class FamilyRead:
    family: str
    title: str
    direction: str               # "bullish" | "bearish" | "neutral" | "elevated" | "compressed" | "n/a"
    score: float                 # signed -1..1 (directional families) or 0..1 magnitude (others)
    intensity_pctile: float      # 0..100, where the window sits vs history
    salient: list                # [(col, title, period_value, pctile, polarity)]
    text: str = ""


@dataclass
class EventHit:
    col: str
    label: str
    count: int
    dates: list                  # ISO strings of when it fired in the window


# magnitude-only families read as elevated/compressed, not bull/bear
_MAGNITUDE_FAMILIES = {"volatility", "liquidity", "social"}
# binary event flags worth surfacing (col -> human label)
_EVENT_FLAGS = {
    "liq_capitulation": "long-liquidation capitulation", "liq_short_panic": "short panic / squeeze",
    "liq_long_spike": "long-liquidation spike", "liq_short_spike": "short-liquidation spike (squeeze)",
    "bs_basis_panic": "basis panic (deleveraging)", "bs_basis_frenzy": "basis frenzy (leveraged greed)",
    "bs_basis_bull_shock": "basis bull shock", "bs_basis_bear_shock": "basis bear shock",
    "fund_sign_flip": "funding sign flip", "s3_smart_extreme_long": "smart-money extreme long",
    "s3_smart_extreme_short": "smart-money extreme short",
}


def _pctile(sorted_ref: np.ndarray, value: float) -> float:
    """Percentile of `value` within a pre-sorted reference array (0..100). NaN-safe via caller."""
    if sorted_ref.size == 0 or not np.isfinite(value):
        return float("nan")
    i = int(np.searchsorted(sorted_ref, value, side="right"))
    return 100.0 * i / sorted_ref.size


def compute_state(df, period_mask, ref_mask=None, max_salient=4):
    """Return (reads: dict[family->FamilyRead], events: list[EventHit], meta: dict)."""
    cols = list(df.columns)
    n = len(df)
    if ref_mask is None:
        ref_mask = np.ones(n, dtype=bool)
    # numpy view of each column once; non-numeric (categorical label) columns -> None (skipped from numeric reads)
    def arr(c):
        if c not in cols:
            return None
        try:
            return np.asarray(df[c].to_numpy(), dtype=np.float64)
        except (ValueError, TypeError):
            return None

    grouped = group_columns(cols)
    reads: dict[str, FamilyRead] = {}

    for fam in FAMILY_ORDER:
        fam_cols = [c for c in grouped.get(fam, []) if c in FEATURES]  # curated, interpretable
        if not fam_cols:
            # still allow a generic read from any norm_ cols in the family
            fam_cols = [c for c in grouped.get(fam, []) if c.lower().startswith(("norm_", "xd_"))][:6]
        if not fam_cols:
            continue
        signed_terms, intens_terms, salient = [], [], []
        for c in fam_cols:
            a = arr(c)
            if a is None:
                continue
            pv = a[period_mask]
            pv = pv[np.isfinite(pv)]
            if pv.size == 0:
                continue
            period_val = float(np.mean(pv))
            ref = a[ref_mask]
            ref = np.sort(ref[np.isfinite(ref)])
            pct = _pctile(ref, period_val)
            if not np.isfinite(pct):
                continue
            feat = FEATURES.get(c)
            pol = feat.polarity if feat else 0
            title = feat.title if feat else c
            # signed contribution: (pct-50)/50 in [-1,1], oriented by polarity
            if pol != 0:
                signed_terms.append(pol * (pct - 50.0) / 50.0)
            intens_terms.append(abs(pct - 50.0) / 50.0 if pol != 0 else pct / 100.0)
            salient.append((c, title, period_val, pct, pol))

        if not salient:
            continue
        score = float(np.mean(signed_terms)) if signed_terms else 0.0
        intensity = float(np.mean(intens_terms)) * 100.0
        # rank salient by deviation from neutral
        salient.sort(key=lambda s: abs(s[3] - 50.0), reverse=True)
        salient = salient[:max_salient]

        if fam in _MAGNITUDE_FAMILIES or not signed_terms:
            # elevated/compressed read
            mean_pct = float(np.mean([s[3] for s in salient]))
            direction = "elevated" if mean_pct >= 65 else ("compressed" if mean_pct <= 35 else "normal")
            score = mean_pct / 100.0
        else:
            direction = "bullish" if score > 0.18 else ("bearish" if score < -0.18 else "neutral")
        reads[fam] = FamilyRead(fam, FAMILIES[fam].title, direction, round(score, 3),
                                round(intensity, 1), salient)

    events = _detect_events(df, period_mask, cols)
    meta = _meta(df, period_mask, cols)
    return reads, events, meta


def _detect_events(df, period_mask, cols) -> list:
    out = []
    idx = np.where(period_mask)[0]
    ts = df["timestamp"].to_numpy() if "timestamp" in cols else None
    import pandas as pd
    for c, label in _EVENT_FLAGS.items():
        if c not in cols:
            continue
        a = np.asarray(df[c].to_numpy(), dtype=np.float64)
        fired = idx[np.nan_to_num(a[idx]) > 0.5]
        if fired.size:
            dates = []
            if ts is not None:
                dates = sorted({str(pd.to_datetime(int(ts[i]), unit="ms").date()) for i in fired})
            out.append(EventHit(c, label, int(fired.size), dates))
    out.sort(key=lambda e: e.count, reverse=True)
    return out


def _meta(df, period_mask, cols) -> dict:
    import pandas as pd
    idx = np.where(period_mask)[0]
    m = {"n_bars": int(idx.size)}
    if "timestamp" in cols and idx.size:
        ts = df["timestamp"].to_numpy()
        m["start"] = str(pd.to_datetime(int(ts[idx[0]]), unit="ms"))
        m["end"] = str(pd.to_datetime(int(ts[idx[-1]]), unit="ms"))
    # precomputed regime labels (modal value over the window)
    for c in ("regime_label", "hurst_regime", "asset_dna"):
        if c in cols and idx.size:
            vals = df[c].to_numpy()[idx]
            try:
                v, cnt = np.unique(vals[~pd.isna(vals)], return_counts=True)
                if v.size:
                    m[c] = str(v[int(np.argmax(cnt))])
            except Exception:
                pass
    # realized move over the window (context only -- NOT a prediction)
    if "close" in cols and idx.size >= 2:
        cl = df["close"].to_numpy().astype(float)
        m["window_return_pct"] = round(100.0 * (cl[idx[-1]] / cl[idx[0]] - 1.0), 2)
        hi = df["high"].to_numpy().astype(float)[idx] if "high" in cols else cl[idx]
        lo = df["low"].to_numpy().astype(float)[idx] if "low" in cols else cl[idx]
        m["window_max_drawup_pct"] = round(100.0 * (np.max(hi) / cl[idx[0]] - 1.0), 2)
        m["window_max_drawdown_pct"] = round(100.0 * (np.min(lo) / cl[idx[0]] - 1.0), 2)
    return m
