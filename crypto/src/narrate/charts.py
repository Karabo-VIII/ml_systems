"""src/narrate/charts.py -- per-chart-type narration + cross-chart comparison.

The same calendar window narrated on DIFFERENT chart types tells different stories:
  - Time bars (1d/4h/1h) sample at fixed clock intervals -- they are the conventional read.
  - Dollar bars sample at fixed traded-value intervals -- they expand during high-activity
    periods and compress during low-activity ones, revealing VALUE-DENSITY structure that
    clock-bars average out.
  - DIB (Dollar Imbalance Bars) form when cumulative signed dollar flow hits a threshold --
    each bar carries strong directional conviction; fewer bars in chop, more in impulsive moves.
  - Range bars form when price range (high-low) hits a threshold -- they strip out time
    entirely and show PRICE-ACTION density, isolating directional extension.

The point of this module is DESCRIPTIVE: what does each chart TYPE let us SEE? Not forecast.
Entry-signal framing only; exits out of scope.

Public API:
    narrate_across_charts(asset, start, end, cadences=(...)) -> CrossChartResult
        .comparison   dict -- structured per-chart reads + agreement analysis
        .prose        str  -- human-readable multi-chart narrative
        .skipped      list -- [(cadence, reason)] for any cadence that raised

    CrossChartResult also has .to_dict() and .to_text() for compatibility.

Design notes:
  - No emoji (Windows cp1252 safety).
  - chart_profile() describes the ANALYTICAL PURPOSE of each chart type -- this is the
    interpretive frame the comparator uses to say WHY a divergence is meaningful.
  - Agreement/divergence is computed on regime direction and trend consensus, not on raw
    numbers, so the prose stays meaningful even when event bars have far more bars than time bars.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Chart-type profiles -- the interpretive frame for each chart type.
# These descriptions inform WHAT each chart reveals and WHY a divergence is meaningful.

CHART_PROFILES = {
    "1d": {
        "family": "time", "name": "Daily (1d)",
        "what_it_sees": (
            "The coarse-grained session view. Each bar is exactly one calendar day regardless of "
            "volume. It captures the overnight/multi-session state that longer-hold swings live in. "
            "Directional signals here are slow, durable, and survive noise. The natural chart for "
            "swing and position-trend setups."
        ),
        "blind_spots": (
            "Volume clustering is invisible -- a day of violent intraday action looks the same as "
            "a quiet day if the close lands near the open. Intraday structure (coils, breakout "
            "attempts, demand zones) is all averaged out."
        ),
    },
    "4h": {
        "family": "time", "name": "4-hour (4h)",
        "what_it_sees": (
            "The intraday structure view. The sweet spot between signal (not too much noise) and "
            "resolution (can see a 12-24h setup unfold). Primary for swing entries: sees the "
            "pullback into a trend, the base formation, the momentum re-ignition."
        ),
        "blind_spots": (
            "Still time-sampled, so a high-volume burst and a low-volume drift both occupy one bar. "
            "Misses very short impulsive moves and is slower than event bars to register value-flow changes."
        ),
    },
    "1h": {
        "family": "time", "name": "Hourly (1h)",
        "what_it_sees": (
            "Intraday momentum and micro-structure. Good for confirming 4h setups at finer "
            "resolution, seeing hourly order-flow dynamics, and reading consolidation ranges."
        ),
        "blind_spots": (
            "High noise relative to signal at daily/4h swing horizons. 1h is the entry-trigger "
            "resolution, not the setup-identification resolution."
        ),
    },
    "30m": {
        "family": "time", "name": "30-minute (30m)",
        "what_it_sees": (
            "Fine intraday structure; useful for timing entries within a 1h-4h setup. Captures "
            "micro-pullbacks and momentum micro-pivots."
        ),
        "blind_spots": (
            "Even more noise than 1h at swing horizons; volume-blind."
        ),
    },
    "15m": {
        "family": "time", "name": "15-minute (15m)",
        "what_it_sees": (
            "Granular intraday texture; near the scalping boundary at swing-trade horizons. "
            "Used for precise entry timing once a higher-timeframe setup is confirmed."
        ),
        "blind_spots": (
            "Very noisy; unreliable for regime reads. Best as an entry-trigger lens only."
        ),
    },
    "dollar": {
        "family": "event", "name": "Dollar bars",
        "what_it_sees": (
            "Each bar closes when a fixed USD value has traded. This equalizes economic "
            "significance: every bar represents the same DOLLAR ACTIVITY. In high-volume "
            "regimes, dollar bars form rapidly (many bars per hour) -- structure is dense. "
            "In low-volume periods, they form slowly -- structure is sparse. Dollar bars "
            "reveal regime intensity that clock bars wash out: a trending impulse with heavy "
            "volume shows many tightly-formed bars with clear directional runs; a ranging "
            "low-volume period shows loose, slow bars."
        ),
        "blind_spots": (
            "The bar count for the same calendar period varies with market activity -- "
            "direct comparison of n_bars to time-bar counts is misleading. Order-flow "
            "and regime reads are computed per-bar, not per-hour, so the absolute intensity "
            "is relative to the asset's own dollar-bar history."
        ),
    },
    "dib": {
        "family": "event", "name": "Dollar imbalance bars (DIB)",
        "what_it_sees": (
            "Each bar closes when the SIGNED dollar imbalance (cumulative buy-vs-sell dollar flow) "
            "hits a threshold. This chart is CONVICTION-WEIGHTED: in directional, high-conviction "
            "moves, bars form quickly (many bars = high conviction flow); in choppy balanced tape, "
            "bar formation is slow (few bars = balanced, indecisive flow). DIB isolates "
            "DIRECTIONAL VALUE FLOW. A strong momentum signal on DIB means buyers/sellers are "
            "persistently dominating dollar flow, not just that price moved."
        ),
        "blind_spots": (
            "Rare on balanced/choppy periods (few bars formed means less statistical weight). "
            "Less useful for volatility or vol-regime reads; strongest for order-flow and "
            "momentum family reads."
        ),
    },
    "range": {
        "family": "event", "name": "Range bars",
        "what_it_sees": (
            "Each bar closes when the high-low PRICE RANGE hits a threshold. Time is stripped out "
            "entirely. This chart reveals PRICE-ACTION DENSITY: how much ground price is covering "
            "per unit of 'move'. In trending regimes, range bars form cleanly in one direction; "
            "in ranges, they flip direction quickly. Range bars filter micro-noise and show the "
            "structural skeleton of price action -- support/resistance and trend clarity are "
            "often sharper than on time bars."
        ),
        "blind_spots": (
            "No volume or time information; order-flow reads are approximate. Cannot say HOW LONG "
            "it took to move -- only that it did move. Volume and activity context is absent."
        ),
    },
    "runs_tick": {
        "family": "event", "name": "Tick runs bars",
        "what_it_sees": "Runs of directional tick prints; microstructure-dense.",
        "blind_spots": "Very high bar count; dollar/value context absent.",
    },
    "runs_volume": {
        "family": "event", "name": "Volume runs bars",
        "what_it_sees": "Runs of volume-weighted directional flow.",
        "blind_spots": "No price-range normalization.",
    },
    "adaptive_vol": {
        "family": "event", "name": "Adaptive volatility bars",
        "what_it_sees": "Bars sized by local volatility -- expands in quiet periods.",
        "blind_spots": "Less common; may have sparse coverage.",
    },
}

_DEFAULT_CADENCES = ("1d", "4h", "1h", "dollar", "dib", "range")

# Families considered most reliable for cross-chart agreement analysis
_AGREEMENT_FAMILIES = ("momentum", "structure", "orderflow")


# ---------------------------------------------------------------------------
@dataclass
class ChartRead:
    """The essential per-chart narration for the comparison layer."""
    cadence: str
    profile_name: str
    profile_family: str           # "time" | "event"
    regime_label: str             # e.g. "trending / up / high-vol"
    direction: str                # "up" | "down" | "sideways"
    vol_state: str                # "elevated" | "normal" | "compressed"
    trending: bool
    trend_score: float
    top_mode: Optional[str]       # first suited_mode key
    salient_reads: list           # [(family, direction, intensity_pctile)]
    n_bars: int
    window_return_pct: Optional[float]
    narration_text: str           # the full per-chart to_text()

    def to_dict(self) -> dict:
        return {
            "cadence": self.cadence, "profile_name": self.profile_name,
            "profile_family": self.profile_family, "regime_label": self.regime_label,
            "direction": self.direction, "vol_state": self.vol_state,
            "trending": self.trending, "trend_score": self.trend_score,
            "top_mode": self.top_mode, "salient_reads": self.salient_reads,
            "n_bars": self.n_bars, "window_return_pct": self.window_return_pct,
        }


@dataclass
class CrossChartResult:
    """Full multi-chart comparison: structured data + prose."""
    asset: str
    start: str
    end: str
    cadences_requested: tuple
    chart_reads: list             # [ChartRead] for succeeded cadences
    skipped: list                 # [(cadence, reason)] for failed cadences
    comparison: dict              # structured agreement/divergence analysis
    prose: str                    # human-readable multi-chart narrative

    def to_dict(self) -> dict:
        return {
            "asset": self.asset, "start": self.start, "end": self.end,
            "cadences_requested": list(self.cadences_requested),
            "chart_reads": [r.to_dict() for r in self.chart_reads],
            "skipped": self.skipped, "comparison": self.comparison,
        }

    def to_text(self) -> str:
        return self.prose


# ---------------------------------------------------------------------------
def narrate_across_charts(
    asset: str,
    start: str = None,
    end: str = None,
    cadences: tuple = _DEFAULT_CADENCES,
) -> CrossChartResult:
    """Narrate (asset, [start, end]) across multiple chart types and compare what each sees.

    Args:
        asset:    e.g. "BTC" or "BTCUSDT"
        start:    ISO date string, e.g. "2025-09-01" (or None = full history)
        end:      ISO date string, e.g. "2025-11-01" (or None = full history)
        cadences: chart types to attempt; missing ones are skipped with reason recorded.

    Returns:
        CrossChartResult with .comparison (dict) and .prose (str).
    """
    from .narrator import narrate

    chart_reads: list[ChartRead] = []
    skipped: list = []

    for cad in cadences:
        try:
            nr = narrate(asset, cadence=cad, start=start, end=end)
            d = nr.to_dict()
            meta = d["meta"]
            regime = meta.get("regime_synthesis", {})
            modes = d["mode_hint"].get("suited_modes", [])
            top_mode = modes[0][0] if modes else None
            salient_reads = [
                (fam, info["direction"], int(round(info.get("intensity_pctile", 0))))
                for fam, info in d["reads"].items()
                if fam in _AGREEMENT_FAMILIES
            ]
            profile = CHART_PROFILES.get(cad, {"family": "event", "name": cad})
            cr = ChartRead(
                cadence=cad,
                profile_name=profile.get("name", cad),
                profile_family=profile.get("family", "event"),
                regime_label=regime.get("label", "unknown"),
                direction=regime.get("direction", "sideways"),
                vol_state=regime.get("vol_state", "normal"),
                trending=regime.get("trending", False),
                trend_score=regime.get("trend_score", 0.0),
                top_mode=top_mode,
                salient_reads=salient_reads,
                n_bars=meta.get("n_bars", 0),
                window_return_pct=meta.get("window_return_pct"),
                narration_text=nr.to_text(),
            )
            chart_reads.append(cr)
        except (ValueError, FileNotFoundError) as exc:
            skipped.append((cad, str(exc)[:160]))
        except Exception as exc:
            skipped.append((cad, f"{type(exc).__name__}: {str(exc)[:140]}"))

    comparison = _build_comparison(chart_reads)
    prose = _render_prose(asset, start, end, chart_reads, skipped, comparison)

    return CrossChartResult(
        asset=asset,
        start=start or "all",
        end=end or "all",
        cadences_requested=tuple(cadences),
        chart_reads=chart_reads,
        skipped=skipped,
        comparison=comparison,
        prose=prose,
    )


# ---------------------------------------------------------------------------
def _build_comparison(chart_reads: list) -> dict:
    """Compute agreement/divergence across chart reads.

    Returns a structured dict with:
      - per_chart: per-cadence summary
      - direction_agreement: do charts agree on direction?
      - trend_agreement: do charts agree on trending vs ranging?
      - vol_agreement: do charts agree on vol state?
      - time_vs_event: does event-bar analysis add anything vs time bars?
      - divergences: list of notable disagreements with interpretive note
      - consensus: the most common direction / mode / vol
    """
    if not chart_reads:
        return {"status": "no_charts_succeeded"}

    time_bars = [r for r in chart_reads if r.profile_family == "time"]
    event_bars = [r for r in chart_reads if r.profile_family == "event"]

    # per-chart summary
    per_chart = {}
    for r in chart_reads:
        per_chart[r.cadence] = {
            "regime_label": r.regime_label,
            "direction": r.direction,
            "vol_state": r.vol_state,
            "trending": r.trending,
            "trend_score": r.trend_score,
            "top_mode": r.top_mode,
            "n_bars": r.n_bars,
            "window_return_pct": r.window_return_pct,
        }

    # direction consensus
    directions = [r.direction for r in chart_reads]
    direction_counts = {}
    for d in directions:
        direction_counts[d] = direction_counts.get(d, 0) + 1
    consensus_direction = max(direction_counts, key=direction_counts.get)
    direction_agreement = len(direction_counts) == 1 or direction_counts.get(consensus_direction, 0) >= len(chart_reads) * 0.6

    # trend/range consensus
    trend_votes = [r.trending for r in chart_reads]
    trending_count = sum(1 for t in trend_votes if t)
    ranging_count = len(trend_votes) - trending_count
    consensus_trending = trending_count >= ranging_count
    trend_agreement = abs(trending_count - ranging_count) >= max(1, len(chart_reads) // 3)

    # vol consensus
    vol_states = [r.vol_state for r in chart_reads]
    vol_counts = {}
    for v in vol_states:
        vol_counts[v] = vol_counts.get(v, 0) + 1
    consensus_vol = max(vol_counts, key=vol_counts.get)
    vol_agreement = vol_counts.get(consensus_vol, 0) >= len(chart_reads) * 0.6

    # mode consensus
    mode_counts = {}
    for r in chart_reads:
        if r.top_mode:
            mode_counts[r.top_mode] = mode_counts.get(r.top_mode, 0) + 1
    consensus_mode = max(mode_counts, key=mode_counts.get) if mode_counts else None

    # time vs event divergences
    divergences = []
    time_vs_event = {"available": bool(time_bars and event_bars)}

    if time_bars and event_bars:
        tb_directions = set(r.direction for r in time_bars)
        eb_directions = set(r.direction for r in event_bars)
        tb_trending = [r.trending for r in time_bars]
        eb_trending = [r.trending for r in event_bars]
        tb_vol = [r.vol_state for r in time_bars]
        eb_vol = [r.vol_state for r in event_bars]

        # direction divergence between time and event
        if tb_directions and eb_directions and not (tb_directions & eb_directions):
            divergences.append({
                "type": "time_vs_event_direction",
                "time_bars": {r.cadence: r.direction for r in time_bars},
                "event_bars": {r.cadence: r.direction for r in event_bars},
                "note": (
                    "Time bars and event bars disagree on direction. Event bars weight by "
                    "value/flow, so this may reflect that the period's ACTIVE trading was "
                    "directionally different from the clock-sampled close-to-close drift. "
                    "Trust the event-bar direction for flow-weighted conviction."
                ),
            })
        elif tb_directions and eb_directions and (tb_directions != eb_directions):
            divergences.append({
                "type": "time_vs_event_direction_partial",
                "time_bar_directions": sorted(tb_directions),
                "event_bar_directions": sorted(eb_directions),
                "note": (
                    "Time bars show mixed direction signals while event bars lean one way (or "
                    "vice versa). The event bars' flow-weighted sampling may be picking up a "
                    "stronger underlying conviction that the clock-sampled bars dilute."
                ),
            })

        # trend/range divergence
        tb_trend_mean = sum(1 for t in tb_trending if t) / max(1, len(tb_trending))
        eb_trend_mean = sum(1 for t in eb_trending if t) / max(1, len(eb_trending))
        if abs(tb_trend_mean - eb_trend_mean) >= 0.5:
            divergences.append({
                "type": "time_vs_event_trend_regime",
                "time_bars_trending_fraction": round(tb_trend_mean, 2),
                "event_bars_trending_fraction": round(eb_trend_mean, 2),
                "note": (
                    "Event bars and time bars disagree on the trending/ranging classification. "
                    "When event bars see a trend but time bars see a range, the active trading "
                    "value is directional even though the clock-sampled drift appears flat -- "
                    "this is the hallmark of a COIL or accumulation phase where big money is "
                    "moving deliberately but price is being contained."
                ),
            })

        # vol divergence
        tb_elevated = sum(1 for v in tb_vol if v == "elevated") / max(1, len(tb_vol))
        eb_elevated = sum(1 for v in eb_vol if v == "elevated") / max(1, len(eb_vol))
        if abs(tb_elevated - eb_elevated) >= 0.4:
            divergences.append({
                "type": "time_vs_event_vol_state",
                "time_bars_elevated_fraction": round(tb_elevated, 2),
                "event_bars_elevated_fraction": round(eb_elevated, 2),
                "note": (
                    "Event bars show a different volatility read than time bars. Dollar/DIB bars "
                    "form more rapidly in high-activity periods -- if event bars show elevated vol "
                    "but daily bars show normal, the activity is concentrated in short bursts "
                    "that the daily bar absorbs. Watch for intraday vol structure."
                ),
            })

        # specific per-event-bar insights
        for r in event_bars:
            for tr in time_bars:
                if r.direction != tr.direction:
                    insight = ""
                    if r.cadence == "dollar":
                        insight = (
                            f"Dollar bars ({r.cadence}: {r.direction}) differ from {tr.cadence} "
                            f"time bars ({tr.direction}). Dollar bars weight by traded USD value, "
                            f"so they may reflect that the high-volume activity was "
                            f"{r.direction}-biased even if clock-sampled returns look {tr.direction}."
                        )
                    elif r.cadence == "dib":
                        insight = (
                            f"DIB ({r.cadence}: {r.direction}) differ from {tr.cadence} bars "
                            f"({tr.direction}). DIB form on conviction-weighted signed flow, so a "
                            f"{r.direction} DIB read means the SIGNED dollar imbalance favored "
                            f"{r.direction} -- aggressive informed flow was {r.direction}."
                        )
                    elif r.cadence == "range":
                        insight = (
                            f"Range bars ({r.cadence}: {r.direction}) differ from {tr.cadence} "
                            f"({tr.direction}). Range bars strip out time; this suggests the "
                            f"structural price movement (measured by range, not by calendar) "
                            f"was {r.direction} even if clock-return was {tr.direction}."
                        )
                    if insight:
                        divergences.append({"type": "cross_chart_direction_detail", "note": insight})
                    break  # one note per event bar is enough

        time_vs_event.update({
            "time_bar_cadences": [r.cadence for r in time_bars],
            "event_bar_cadences": [r.cadence for r in event_bars],
            "direction_aligned": not any(d["type"] in ("time_vs_event_direction", "time_vs_event_direction_partial")
                                         for d in divergences),
            "regime_aligned": not any(d["type"] == "time_vs_event_trend_regime" for d in divergences),
        })

    # timeframe-within-time-bars divergences (1d vs faster time bars)
    if len(time_bars) >= 2:
        daily = next((r for r in time_bars if r.cadence == "1d"), None)
        fast_time = [r for r in time_bars if r.cadence in ("4h", "1h", "30m", "15m")]
        if daily and fast_time:
            disagreeing_fast = [r for r in fast_time if r.direction != daily.direction]
            if len(disagreeing_fast) >= len(fast_time) * 0.5:
                divergences.append({
                    "type": "htf_vs_ltf_direction",
                    "daily_direction": daily.direction,
                    "fast_time_bar_directions": {r.cadence: r.direction for r in fast_time},
                    "note": (
                        "Daily bars and faster time bars disagree on direction. This is a CLASSIC "
                        "structure divergence: the daily trend has one bias, but intraday structure "
                        "is pushing the other way. Watch for the lower-timeframe to resolve into "
                        "the daily direction (trend continuation) or break it (potential reversal)."
                    ),
                })

    # no divergences found
    if not divergences:
        divergences.append({
            "type": "full_agreement",
            "note": (
                "All chart types agree on direction and regime. Cross-chart confirmation "
                "strengthens the read: the signal is consistent regardless of how time/value "
                "is sliced."
            ),
        })

    return {
        "n_charts": len(chart_reads),
        "per_chart": per_chart,
        "direction_agreement": direction_agreement,
        "trend_agreement": trend_agreement,
        "vol_agreement": vol_agreement,
        "consensus_direction": consensus_direction,
        "consensus_trending": consensus_trending,
        "consensus_vol": consensus_vol,
        "consensus_mode": consensus_mode,
        "time_vs_event": time_vs_event,
        "divergences": divergences,
    }


# ---------------------------------------------------------------------------
def _render_prose(
    asset: str,
    start: Optional[str],
    end: Optional[str],
    chart_reads: list,
    skipped: list,
    comparison: dict,
) -> str:
    """Render the multi-chart comparison as a crisp, honest prose narrative."""
    if not chart_reads:
        skip_reasons = "; ".join(f"{cad}: {reason[:60]}" for cad, reason in skipped[:5])
        return (
            f"No chart reads succeeded for {asset} [{start} -> {end}]. "
            f"Cadences tried: {skip_reasons or 'none'}."
        )

    L = []
    period_str = f"[{start} -> {end}]" if start or end else "[full history]"
    L.append(f"## {asset} -- multi-chart read -- {period_str}")
    L.append(f"({len(chart_reads)} chart type(s) succeeded; {len(skipped)} skipped)")
    L.append("")

    # consensus headline
    c = comparison
    consensus_dir = c.get("consensus_direction", "?")
    consensus_mode = c.get("consensus_mode", "?")
    consensus_vol = c.get("consensus_vol", "normal")
    trending_str = "trending" if c.get("consensus_trending") else "ranging"
    vol_str = (f" with {consensus_vol} volatility") if consensus_vol != "normal" else ""

    charts_str = ", ".join(r.cadence for r in chart_reads)
    L.append(f"Across charts ({charts_str}), {asset} reads as **{trending_str} / {consensus_dir}{vol_str}**. "
             f"The consensus setup mode is **{consensus_mode or 'unclear'}**.")
    L.append("")

    # per-chart summary table (brief)
    L.append("**Per-chart regime summary:**")
    for r in chart_reads:
        mode_note = f", mode={r.top_mode}" if r.top_mode else ""
        ret_note = (f", realized return {r.window_return_pct:+.1f}%" if r.window_return_pct is not None else "")
        what = CHART_PROFILES.get(r.cadence, {}).get("what_it_sees", "")
        what_short = what[:90].rstrip() + "..." if len(what) > 90 else what
        L.append(f"- **{r.profile_name}** ({r.n_bars:,} bars): {r.regime_label}{mode_note}{ret_note}")
        L.append(f"  [What it sees: {what_short}]")
    L.append("")

    # time vs event comparison
    tve = c.get("time_vs_event", {})
    if tve.get("available"):
        tb_cads = tve.get("time_bar_cadences", [])
        eb_cads = tve.get("event_bar_cadences", [])
        dir_aligned = tve.get("direction_aligned", True)
        reg_aligned = tve.get("regime_aligned", True)

        if dir_aligned and reg_aligned:
            L.append(
                f"**Time bars ({', '.join(tb_cads)}) and event bars ({', '.join(eb_cads)}) AGREE** "
                f"on both direction and trend/range regime. The cross-chart confirmation is clean: "
                f"whether you sample by clock, by dollar value, or by flow conviction, the read is "
                f"the same. No chart-type advantage is detectable in this window."
            )
        elif not dir_aligned:
            L.append(
                f"**Time bars ({', '.join(tb_cads)}) and event bars ({', '.join(eb_cads)}) DIVERGE on direction.** "
                f"This is the most informative divergence: event bars weight by dollar value / "
                f"flow conviction, so they may reveal that the ACTIVE trading within the period "
                f"had a different directional bias than the clock-sampled price drift. "
                f"Investigate which chart type best matches the period's actual order-flow regime."
            )
        elif not reg_aligned:
            L.append(
                f"**Event bars ({', '.join(eb_cads)}) and time bars ({', '.join(tb_cads)}) disagree "
                f"on trending vs ranging regime.** This suggests the active-value periods have a "
                f"different structure from the clock-sampled view -- possible coil / accumulation "
                f"where large money moves deliberately while clock returns look flat."
            )
        L.append("")

    # divergences
    divs = c.get("divergences", [])
    noteworthy = [d for d in divs if d.get("type") != "full_agreement"]
    if noteworthy:
        L.append("**Notable divergences across chart types:**")
        for d in noteworthy[:4]:
            L.append(f"- {d['note']}")
        L.append("")
    else:
        L.append(
            "**Cross-chart agreement:** All chart types agree (no material divergences found). "
            "The read is robust to chart-type sampling choice."
        )
        L.append("")

    # htf timeframe ladder note (if multiple time bars available)
    time_reads = {r.cadence: r for r in chart_reads if r.profile_family == "time"}
    if "1d" in time_reads and "4h" in time_reads:
        d1 = time_reads["1d"]
        h4 = time_reads["4h"]
        if d1.direction == h4.direction:
            L.append(
                f"Daily and 4h bars are **aligned ({d1.direction})** -- the timeframe ladder is "
                f"coherent. A swing entry in the direction of the trend has timeframe confluence."
            )
        else:
            L.append(
                f"Daily ({d1.direction}) and 4h ({h4.direction}) disagree -- the timeframe "
                f"ladder is in CONFLICT. Avoid directional swing entries until the lower "
                f"timeframe resolves into the daily direction or the daily trend changes."
            )
        L.append("")

    # skipped cadences note
    if skipped:
        L.append("**Cadences not available for this asset/period:**")
        for cad, reason in skipped:
            short = reason[:100] + "..." if len(reason) > 100 else reason
            L.append(f"- {cad}: {short}")
        L.append("")

    # framing reminder
    L.append(
        "_Framing: This is a DESCRIPTIVE multi-chart read (the 'what'). Each chart type reveals "
        "a different structural lens on the same calendar window. Entry-signal framing only -- "
        "exits are a separate domain (out of scope). Per-setup, not per-candle._"
    )

    return "\n".join(L)


# ---------------------------------------------------------------------------
def demo(asset: str = "BTC", start: str = "2025-09-01", end: str = "2025-11-01",
         cadences: tuple = _DEFAULT_CADENCES, verbose: bool = True) -> CrossChartResult:
    """CLI-usable demo: run narrate_across_charts and print the comparison + prose.

    Usage (module):
        python -m narrate.charts
        python -m narrate.charts --asset BTC --start 2025-09-01 --end 2025-11-01

    Usage (import):
        from narrate.charts import demo
        result = demo("BTC", "2025-09-01", "2025-11-01")
    """
    import json

    result = narrate_across_charts(asset, start, end, cadences=cadences)

    if verbose:
        print("=" * 72)
        print(f"CROSS-CHART COMPARISON: {asset} [{start} -> {end}]")
        print("=" * 72)
        print()
        print("--- STRUCTURED COMPARISON ---")
        comp_summary = {k: v for k, v in result.comparison.items() if k != "per_chart"}
        print(json.dumps(comp_summary, indent=2, default=str))
        print()
        print("--- PER-CHART REGIME SUMMARY ---")
        for r in result.chart_reads:
            print(f"  {r.cadence:12s}  regime={r.regime_label:<35s}  n_bars={r.n_bars:>7,}  "
                  f"direction={r.direction:<9s}  trending={str(r.trending):<5s}  "
                  f"top_mode={r.top_mode or '-'}")
        if result.skipped:
            print()
            print("--- SKIPPED ---")
            for cad, reason in result.skipped:
                print(f"  {cad}: {reason[:90]}")
        print()
        print("--- PROSE MULTI-CHART READ ---")
        print(result.prose)

    return result


if __name__ == "__main__":
    import sys

    def _parse_args():
        asset, start, end = "BTC", "2025-09-01", "2025-11-01"
        args = sys.argv[1:]
        for i, a in enumerate(args):
            if a == "--asset" and i + 1 < len(args):
                asset = args[i + 1]
            elif a == "--start" and i + 1 < len(args):
                start = args[i + 1]
            elif a == "--end" and i + 1 < len(args):
                end = args[i + 1]
        return asset, start, end

    _asset, _start, _end = _parse_args()
    demo(_asset, _start, _end)
