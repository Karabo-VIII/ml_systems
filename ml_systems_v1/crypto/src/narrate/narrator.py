"""src/narrate/narrator.py -- the combined market-narration engine.

narrate(asset, cadence, start, end) -> MarketNarration: loads chimera for the (asset, chart-type), decomposes the
window into family reads (state.compute_state), layers crypto-specific caveats, optionally augments with a TS
foundation model (MOMENT) and our own trained artifacts, and renders BOTH a structured object and a plain-language
narration of "the what".

Strictly DESCRIPTIVE + ENTRY-FRAMED: it characterizes state and names which strategy MODE the state suits; it does
NOT forecast price and says NOTHING about exits (a separate domain).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import crypto_context
from .feature_map import FAMILIES, coverage_report
from .state import FamilyRead, compute_state
from .strategy_archetypes import select_for


@dataclass
class MarketNarration:
    asset: str
    cadence: str
    meta: dict
    reads: dict                  # family -> FamilyRead
    events: list                 # EventHit
    crypto_caveats: list
    mode_hint: dict              # which strategy MODE the state suits
    foundation: dict = field(default_factory=dict)   # MOMENT / artifact layers (optional)
    coverage: dict = field(default_factory=dict)

    # ---- rendering
    def to_text(self) -> str:
        return render_prose(self)

    def to_dict(self) -> dict:
        return {
            "asset": self.asset, "cadence": self.cadence, "meta": self.meta,
            "reads": {k: {"direction": r.direction, "score": r.score, "intensity_pctile": r.intensity_pctile,
                          "salient": [(c, t, round(v, 3), round(p, 1)) for (c, t, v, p, pol) in r.salient]}
                      for k, r in self.reads.items()},
            "events": [{"flag": e.col, "label": e.label, "count": e.count, "dates": e.dates} for e in self.events],
            "crypto_caveats": self.crypto_caveats, "mode_hint": self.mode_hint,
            "foundation": self.foundation, "coverage": self.coverage,
        }


def _regime_synthesis(reads, meta) -> dict:
    """Classify the window's regime (trend vs range, risk-on/off) from the reads -> drives the mode hint."""
    structure = reads.get("structure")
    momentum = reads.get("momentum")
    vol = reads.get("volatility")
    # trending if structure says efficient/directional and momentum is non-neutral
    trend_score = 0.0
    if momentum:
        trend_score += abs(momentum.score)
    if structure and structure.direction in ("bullish", "bearish"):
        trend_score += abs(structure.score)
    trending = trend_score >= 0.45
    direction = "up" if (momentum and momentum.score > 0) else ("down" if (momentum and momentum.score < 0) else "sideways")
    vol_state = vol.direction if vol else "normal"
    label = (f"{'trending' if trending else 'ranging'} / {direction}"
             f"{' / high-vol' if vol_state == 'elevated' else (' / compressed' if vol_state == 'compressed' else '')}")
    return {"trending": trending, "direction": direction, "vol_state": vol_state, "label": label,
            "trend_score": round(trend_score, 2)}


def _mode_hint(regime, reads) -> dict:
    """Map the regime to the strategy MODE that suits it (per the master map). Entry-framed, no exit."""
    sel = select_for()
    suited = []
    if regime["trending"]:
        suited.append(("swing", "trending multi-day MOVE -- the primary mode"))
        if regime["vol_state"] == "compressed":
            suited.append(("breakout", "compressed vol in a trend = coil; expansion-break entries"))
        else:
            suited.append(("intraday_momentum", "trend with live vol -- pullback-into-trend / momentum-with-confirmation"))
    else:
        if regime["vol_state"] == "compressed":
            suited.append(("breakout", "ranging + compressed vol = a coil; watch for the expansion break"))
        suited.append(("mean_reversion", "ranging regime -- fade extremes back to the mean (regime-gated only)"))
    # liquidation/positioning extremes -> event-driven entry layer
    liq = reads.get("liquidation")
    if liq and any(e for e in ()):  # placeholder; events handled in prose
        pass
    return {"suited_modes": suited, "primary_mandate_mode": sel["primary_mode"],
            "avoid": sel["misfit_traps"], "note": "Entry-signal framing only; exit is a separate domain (out of scope)."}


def narrate(asset, cadence="4h", start=None, end=None, with_foundation=False, with_artifacts=False) -> MarketNarration:
    """Narrate the 'what' of (asset, cadence) over [start, end]. start/end are ISO date strings or None (full history)."""
    import pandas as pd
    from pipeline.chimera_loader import ChimeraLoader

    sym = asset if asset.upper().endswith(("USDT", "USD")) else asset.upper() + "USDT"
    g = ChimeraLoader().load(sym, cadence=cadence)
    cols = list(g.columns)
    ts = g["timestamp"].to_numpy().astype(np.int64)
    # ensure sorted
    if not np.all(np.diff(ts) > 0):
        order = np.argsort(ts, kind="stable")
        g = g[order]; ts = ts[order]
    n = len(ts)

    def ms(x):
        return int(pd.Timestamp(x).value // 1_000_000)
    lo = ms(start) if start else ts[0]
    hi = ms(end) if end else ts[-1] + 1
    period_mask = (ts >= lo) & (ts < hi)
    if period_mask.sum() == 0:
        raise ValueError(f"no {sym} {cadence} bars in [{start}, {end}] (have {pd.to_datetime(ts[0],unit='ms').date()}"
                         f" .. {pd.to_datetime(ts[-1],unit='ms').date()})")
    ref_mask = np.ones(n, dtype=bool)  # compare the window against the asset's full history

    # convert polars -> a column-access shim compute_state expects (df[col].to_numpy())
    df = _PolarsShim(g)
    reads, events, meta = compute_state(df, period_mask, ref_mask)
    meta["symbol"] = sym

    regime = _regime_synthesis(reads, meta)
    meta["regime_synthesis"] = regime
    mode_hint = _mode_hint(regime, reads)
    active_fams = list(reads.keys())
    caveats = crypto_context.caveats_for_families(active_fams)
    cov = coverage_report(cols)

    foundation = {}
    if with_foundation:
        try:
            from .foundation import foundation_layer
            foundation["moment"] = foundation_layer(df, period_mask, ref_mask)
        except Exception as e:
            foundation["moment_error"] = str(e)[:160]
    if with_artifacts:
        try:
            from .artifacts import artifact_layer
            foundation["artifacts"] = artifact_layer(sym, cadence, df, period_mask)
        except Exception as e:
            foundation["artifacts_error"] = str(e)[:160]

    return MarketNarration(asset=sym, cadence=cadence, meta=meta, reads=reads, events=events,
                           crypto_caveats=caveats, mode_hint=mode_hint, foundation=foundation, coverage=cov)


class _PolarsShim:
    """Minimal df[col].to_numpy() shim so state.py works on a polars frame without a pandas conversion of 243 cols."""
    def __init__(self, g):
        self._g = g
        self.columns = list(g.columns)

    def __getitem__(self, col):
        return _Col(self._g[col])

    def __len__(self):
        return len(self._g)


class _Col:
    def __init__(self, s):
        self._s = s

    def to_numpy(self):
        return self._s.to_numpy()


# ---------------------------------------------------------------------------
def render_prose(nr: MarketNarration) -> str:
    m, reads, regime = nr.meta, nr.reads, nr.meta.get("regime_synthesis", {})
    L = []
    period = f"{m.get('start','?')[:10]} -> {m.get('end','?')[:10]} ({m.get('n_bars','?')} {nr.cadence} bars)"
    L.append(f"## {nr.asset} -- {nr.cadence} chart -- {period}")
    # context line (realized move = CONTEXT, not prediction)
    if "window_return_pct" in m:
        L.append(f"Over the window price moved {m['window_return_pct']:+.1f}% "
                 f"(max drawup {m.get('window_max_drawup_pct','?')}%, max drawdown {m.get('window_max_drawdown_pct','?')}%). "
                 f"[realized context, not a forecast]")
    # regime headline
    L.append(f"\n**Regime:** {regime.get('label','?')}"
             + (f"  |  pipeline labels: {m.get('regime_label','-')}, hurst={m.get('hurst_regime','-')}, "
                f"dna={m.get('asset_dna','-')}" if any(k in m for k in ('regime_label','hurst_regime','asset_dna')) else ""))

    # the family reads, in reading order
    def line(fam, verb_map):
        r = reads.get(fam)
        if not r:
            return None
        sal = ", ".join(f"{t} (p{int(p)})" for (_c, t, _v, p, _pol) in r.salient[:3])
        v = verb_map.get(r.direction, r.direction)
        return f"- **{FAMILIES[fam].title}:** {v} (intensity p{int(r.intensity_pctile)}). {sal}."

    L.append("\n**The decomposition:**")
    for fam, vm in [
        ("structure", {"bullish": "price is extended ABOVE its structure", "bearish": "price sits BELOW its structure",
                       "neutral": "price is mid-structure"}),
        ("momentum", {"bullish": "momentum is positive / building", "bearish": "momentum is negative / fading",
                      "neutral": "momentum is flat"}),
        ("volatility", {"elevated": "volatility is EXPANDED / active", "compressed": "volatility is COMPRESSED (a coil)",
                        "normal": "volatility is normal"}),
        ("orderflow", {"bullish": "aggressive BUYERS control the tape", "bearish": "aggressive SELLERS control the tape",
                       "neutral": "order flow is balanced", "elevated": "flow intensity is high"}),
        ("liquidity", {"elevated": "the book is deep/stable", "compressed": "the book is THIN/fragile", "normal": "book is normal"}),
        ("derivatives", {"bullish": "leveraged positioning leans LONG (funding/basis rich)",
                         "bearish": "leveraged positioning leans SHORT (funding/basis cheap)", "neutral": "funding/basis are balanced"}),
        ("liquidation", {"bullish": "forced flow favors squeezes UP", "bearish": "forced SELLING (down-cascade pressure)",
                         "neutral": "no notable forced flow"}),
        ("positioning", {"bullish": "accounts lean long (smart-money tilt)", "bearish": "accounts lean short / crowd-long fade-risk",
                         "neutral": "positioning is balanced"}),
        ("whale", {"bullish": "whales are net BUYING", "bearish": "whales are net SELLING", "neutral": "whale flow is flat"}),
        ("cross_asset", {"bullish": "BTC/market beta is supportive + this asset ranks strong",
                         "bearish": "BTC/market beta is a headwind / this asset lags", "neutral": "cross-asset context is mixed"}),
        ("social", {"elevated": "retail attention is RISING", "compressed": "attention is quiet", "normal": "attention is normal"}),
    ]:
        ln = line(fam, vm)
        if ln:
            L.append(ln)

    # events
    if nr.events:
        L.append("\n**Notable events in the window:**")
        for e in nr.events[:8]:
            d = (" -- " + ", ".join(e.dates[:4]) + ("..." if e.count > 4 else "")) if e.dates else ""
            L.append(f"- {e.label}: {e.count}x{d}")

    # mode hint (entry-framed)
    L.append("\n**Which mode this state suits (entry-signal framing; exit is a separate domain):**")
    for k, why in nr.mode_hint["suited_modes"]:
        L.append(f"- {k}: {why}")
    L.append(f"- avoid: {', '.join(nr.mode_hint['avoid'])} (per-candle / infra traps)")

    # foundation augmentation
    if nr.foundation:
        L.append("\n**Foundation-model / artifact augmentation:**")
        mo = nr.foundation.get("moment")
        if mo:
            L.append(f"- MOMENT: anomaly p{mo.get('anomaly_pctile','?')} ({mo.get('anomaly_read','')}); "
                     f"nearest historical analog: {mo.get('analog','n/a')}; validation: {mo.get('validation','n/a')}")
        if "moment_error" in nr.foundation:
            L.append(f"- MOMENT unavailable: {nr.foundation['moment_error']}")
        art = nr.foundation.get("artifacts")
        if art:
            L.append(f"- our artifacts: {art}")

    # crypto caveats
    if nr.crypto_caveats:
        L.append("\n**Crypto-context caveats (why this reads differently than equities/FX):**")
        for c in nr.crypto_caveats[:6]:
            L.append(f"- {c}")

    L.append(f"\n_Coverage: {nr.coverage.get('n_curated','?')}/{nr.coverage.get('n_tradeable','?')} tradeable "
             f"chimera columns curated ({nr.coverage.get('curated_pct','?')}%), "
             f"{len(nr.coverage.get('families_present',{}))} families._")
    return "\n".join(L)
