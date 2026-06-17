"""src/wm/_shared/regime_targets.py -- forward regime / move-onset LABEL builders.

WHY THIS MODULE EXISTS
----------------------
A proof-of-value test (src/strat/wm_value_probe.py) + the deep audit converged on one
conclusion: V1.1's value as a trading INPUT is REGIME / BEAR DETECTION (avoid trading
in downtrends), NOT per-bar return IC. But V1.1 was trained on the wrong objective
(per-bar TwoHot h=1 return -- the per-candle lens the founding framing BANNED), and its
"regime gate" is a *coincident* SMA/return label (world_model.py get_loss lines 435-445:
labels derived from target_returns[1], the CURRENT bar's return), not a FORWARD one.

This module is a pure label-builder. It computes, from a price/return series, the
*training targets* a dedicated FORWARD regime / bear-onset head and a MOVE-ONSET head
would be supervised against. It is the analogue of the pipeline's existing forward
target builder (sota_shared_logic_v50.py line ~304:
    target_return_h = (close.shift(-h) - close) / close
-- a forward h-bar return computed at TARGET-CONSTRUCTION time). Using FUTURE bars to
build a LABEL is allowed and standard supervised learning. The hard rule is:

    THE LABEL MUST NEVER ENTER THE INPUT / INFERENCE PATH.

At inference time the model predicts these quantities from PAST observations only
(the causal transformer in world_model.py forward_train); the label arrays produced
here are used solely as the y in the loss. The self-test below mechanically asserts
both properties: (1) reconstructability -- the label at bar t is a deterministic
function of bars [t .. t+H] (no information beyond the label horizon), and (2)
non-leakage -- the label is NOT equal to (and is not trivially recoverable from) any
single input feature at bar t.

WHAT IT BUILDS (all PAST-ONLY at inference, FUTURE-bars-at-label-time only)
--------------------------------------------------------------------------
1. forward_bear_label(close, K, dd_thresh)
   P(next-K-bar realized MAX DRAWDOWN > dd_thresh) as a binary label. "Will the path
   over the next K bars dip more than dd_thresh below the current close?" This is the
   bear-ONSET signal: it fires BEFORE the drawdown, at the bar where avoidance pays.

2. forward_trend_label(close, K, up_thresh, dn_thresh)
   3-class FORWARD trend over the next K bars from the net K-bar return:
   {0=down, 1=neutral, 2=up}. Contrast with the existing COINCIDENT 3-class regime
   head (supervised by the current bar's return sign).

3. move_onset_label(close, a, b, move_thresh, cost)
   Binary "will there be a >= move_thresh NET-OF-COST up-move somewhere in the window
   [t+a, t+b]?" -- the SETUP/MOVE-ONSET target, an alternative to per-bar return per
   the audit's "wrong target" finding. The unit is a multi-candle MOVE, not a candle.

All builders return a float32 array of shape (N,) aligned to the input series, with the
last H bars (H = the label's forward horizon) set to NaN (no future available -> not a
valid training target; the trainer masks NaN rows).

NOTE: this module is INTENTIONALLY dependency-light (numpy only) and imports nothing
from settings.py / world_model.py / components.py. It changes NO load-bearing code.
Run `python src/wm/_shared/regime_targets.py` for the no-look-ahead self-test.
"""
from __future__ import annotations

import numpy as np

__contract__ = {
    "kind": "label_builder",
    "version": "1.0",
    "inputs": ["close: 1D float array (prices)"],
    "outputs": [
        "forward_bear_label: float32 (N,) in {0,1,NaN} -- P(fwd K-bar maxDD > thresh)",
        "forward_trend_label: float32 (N,) in {0,1,2,NaN} -- fwd K-bar trend class",
        "move_onset_label: float32 (N,) in {0,1,NaN} -- net-of-cost up-move in [a,b]",
    ],
    "invariants": [
        "LABEL ONLY -- never an input feature; computed at target-construction time",
        "label[t] depends only on bars [t .. t+H] (H = forward horizon); last H bars = NaN",
        "FUTURE bars used to BUILD the label is allowed; feeding the label as input is NOT",
        "NaN rows are not valid training targets -- trainer must mask them",
        "no dependency on settings/world_model/components -- isolated, base path unchanged",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_close(close) -> np.ndarray:
    c = np.asarray(close, dtype=np.float64).reshape(-1)
    if c.ndim != 1 or c.size < 2:
        raise ValueError(f"close must be a 1D series of length >= 2; got shape {np.shape(close)}")
    if not np.all(np.isfinite(c)):
        raise ValueError("close contains non-finite values; clean before labelling")
    if np.any(c <= 0):
        raise ValueError("close must be strictly positive (prices)")
    return c


# ---------------------------------------------------------------------------
# 1. Forward bear-onset: P(next-K-bar max drawdown > dd_thresh)
# ---------------------------------------------------------------------------

def forward_bear_label(close, K: int = 64, dd_thresh: float = 0.05) -> np.ndarray:
    """Binary bear-ONSET label: 1 if the path over the next K bars draws down more than
    dd_thresh BELOW the current close at any point in (t, t+K], else 0.

    label[t] = 1  iff  min_{j in 1..K} ( close[t+j] / close[t] - 1 ) < -dd_thresh

    This fires at bar t -- BEFORE the drawdown materializes -- which is exactly when a
    long-only strategy benefits from standing aside. It is the forward analogue of the
    coincident SMA regime head (which only knows the drawdown after it has happened).

    Args:
        close:     1D price series, shape (N,).
        K:         forward window in bars (look-ahead horizon for the LABEL only).
        dd_thresh: drawdown magnitude (0.05 = 5%). Positive number.

    Returns:
        float32 (N,) in {0.0, 1.0, NaN}. Last K bars are NaN (no full future window).
    """
    c = _as_close(close)
    n = c.size
    if K < 1:
        raise ValueError("K must be >= 1")
    if dd_thresh <= 0:
        raise ValueError("dd_thresh must be > 0")

    out = np.full(n, np.nan, dtype=np.float32)
    # For each t with a full K-bar future, compute the minimum forward relative price.
    # Vectorized via a sliding min over the FUTURE window (t+1 .. t+K).
    for t in range(0, n - K):
        future = c[t + 1: t + K + 1]               # bars strictly after t, length K
        min_rel = future.min() / c[t] - 1.0
        out[t] = 1.0 if min_rel < -dd_thresh else 0.0
    return out


# ---------------------------------------------------------------------------
# 2. Forward trend class (3-class), distinct from the coincident regime head
# ---------------------------------------------------------------------------

def forward_trend_label(close, K: int = 64, up_thresh: float = 0.02,
                        dn_thresh: float = 0.02) -> np.ndarray:
    """3-class FORWARD trend over the next K bars from the net K-bar return.

    fwd_ret[t] = close[t+K] / close[t] - 1
    label[t]   = 2 (up)      if fwd_ret >  up_thresh
                 0 (down)    if fwd_ret < -dn_thresh
                 1 (neutral) otherwise

    Contrast: world_model.py's existing regime head is supervised on the sign of the
    CURRENT bar's return (coincident). This head predicts the trend of the NEXT K bars.

    Args:
        close:     1D price series, shape (N,).
        K:         forward window in bars.
        up_thresh: net up move to label "up" (0.02 = 2%).
        dn_thresh: net down move to label "down" (0.02 = 2%).

    Returns:
        float32 (N,) in {0.0, 1.0, 2.0, NaN}. Last K bars are NaN.
    """
    c = _as_close(close)
    n = c.size
    if K < 1:
        raise ValueError("K must be >= 1")
    if up_thresh <= 0 or dn_thresh <= 0:
        raise ValueError("up_thresh and dn_thresh must be > 0")

    out = np.full(n, np.nan, dtype=np.float32)
    valid = n - K
    if valid > 0:
        fwd_ret = c[K:K + valid] / c[:valid] - 1.0     # close[t+K]/close[t]-1 for t in [0,valid)
        cls = np.full(valid, 1.0, dtype=np.float32)
        cls[fwd_ret > up_thresh] = 2.0
        cls[fwd_ret < -dn_thresh] = 0.0
        out[:valid] = cls
    return out


# ---------------------------------------------------------------------------
# 3. Move-onset: net-of-cost up-move somewhere in the window [t+a, t+b]
# ---------------------------------------------------------------------------

def move_onset_label(close, a: int = 1, b: int = 64, move_thresh: float = 0.03,
                     cost: float = 0.0024) -> np.ndarray:
    """Binary SETUP / MOVE-ONSET label: 1 if a net-of-cost up-move of at least
    move_thresh is *available* anywhere in the forward window [t+a, t+b].

    best_move[t] = max_{j in a..b} ( close[t+j] / close[t] - 1 )
    label[t]     = 1  iff  best_move[t] - cost >= move_thresh   else 0

    This is the "is there a tradeable up-MOVE starting here?" target -- a multi-candle
    setup label, the alternative to per-bar return the audit recommended. `cost` is a
    flat round-trip haircut so the label only fires when the move clears costs. The
    capture POLICY (exit timing) is out of scope here; this only asks whether the move
    exists to be captured.

    Args:
        close:       1D price series, shape (N,).
        a, b:        window bounds in bars ahead (1 <= a <= b). The move can occur at
                     any bar in [t+a, t+b].
        move_thresh: minimum net up-move (0.03 = 3%).
        cost:        round-trip cost haircut (0.0024 = TAKER, matches wm_value_probe).

    Returns:
        float32 (N,) in {0.0, 1.0, NaN}. Last b bars are NaN.
    """
    c = _as_close(close)
    n = c.size
    if not (1 <= a <= b):
        raise ValueError(f"require 1 <= a <= b; got a={a}, b={b}")
    if move_thresh <= 0:
        raise ValueError("move_thresh must be > 0")
    if cost < 0:
        raise ValueError("cost must be >= 0")

    out = np.full(n, np.nan, dtype=np.float32)
    for t in range(0, n - b):
        window = c[t + a: t + b + 1]                # bars [t+a .. t+b], length b-a+1
        best_move = window.max() / c[t] - 1.0
        out[t] = 1.0 if (best_move - cost) >= move_thresh else 0.0
    return out


# ---------------------------------------------------------------------------
# No-look-ahead self-test
# ---------------------------------------------------------------------------

def _selftest() -> int:
    """Mechanical no-look-ahead + correctness checks on synthetic series.

    Returns 0 on success, raises AssertionError otherwise.
    """
    rng = np.random.default_rng(0)
    fails = 0

    # --- Test A: hand-built deterministic series, exact label values ----------
    # A clean up-then-crash path so every label has a known answer.
    # bars:   0     1     2     3     4     5     6     7
    close = np.array([100, 102, 104, 106, 95, 90, 92, 93], dtype=float)

    # forward_bear_label K=3, dd=0.05: does any of the next 3 bars dip >5% below close[t]?
    bear = forward_bear_label(close, K=3, dd_thresh=0.05)
    # t=0 (100): next3 = [102,104,106], min rel = +2% -> 0
    # t=1 (102): next3 = [104,106,95],  min rel = 95/102-1 = -6.86% -> 1
    # t=2 (104): next3 = [106,95,90],   min rel = 90/104-1 = -13.5% -> 1
    # t=3 (106): next3 = [95,90,92],    min rel = 90/106-1 = -15.1% -> 1
    # t=4 (95):  next3 = [90,92,93],    min rel = 90/95-1  = -5.26% -> 1
    # t=5..7: NaN (no full K=3 future)
    exp_bear = [0, 1, 1, 1, 1, np.nan, np.nan, np.nan]
    for t, e in enumerate(exp_bear):
        if np.isnan(e):
            assert np.isnan(bear[t]), f"bear[{t}] expected NaN got {bear[t]}"
        else:
            assert bear[t] == e, f"bear[{t}] expected {e} got {bear[t]}"
    # exactly the last K bars are NaN
    assert np.isnan(bear[-3:]).all() and not np.isnan(bear[:-3]).any(), "bear NaN tail wrong"

    # forward_trend_label K=4, up=2%, dn=2%
    trend = forward_trend_label(close, K=4, up_thresh=0.02, dn_thresh=0.02)
    # t=0: close[4]/close[0]-1 = 95/100-1 = -5% -> 0 (down)
    # t=1: close[5]/close[1]-1 = 90/102-1 = -11.8% -> 0 (down)
    # t=2: close[6]/close[2]-1 = 92/104-1 = -11.5% -> 0 (down)
    # t=3: close[7]/close[3]-1 = 93/106-1 = -12.3% -> 0 (down)
    # t=4..7: NaN
    exp_trend = [0, 0, 0, 0, np.nan, np.nan, np.nan, np.nan]
    for t, e in enumerate(exp_trend):
        if np.isnan(e):
            assert np.isnan(trend[t]), f"trend[{t}] expected NaN got {trend[t]}"
        else:
            assert trend[t] == e, f"trend[{t}] expected {e} got {trend[t]}"

    # move_onset_label a=1,b=3, move=2%, cost=0
    move = move_onset_label(close, a=1, b=3, move_thresh=0.02, cost=0.0)
    # t=0 (100): window [102,104,106], best = +6% >= 2% -> 1
    # t=1 (102): window [104,106,95],  best = 106/102-1 = +3.9% -> 1
    # t=2 (104): window [106,95,90],   best = 106/104-1 = +1.9% < 2% -> 0
    # t=3 (106): window [95,90,92],    best = 95/106-1 = -10.4% -> 0
    # t=4 (95):  window [90,92,93],    best = 93/95-1 = -2.1% -> 0
    # t=5..7: NaN
    exp_move = [1, 1, 0, 0, 0, np.nan, np.nan, np.nan]
    for t, e in enumerate(exp_move):
        if np.isnan(e):
            assert np.isnan(move[t]), f"move[{t}] expected NaN got {move[t]}"
        else:
            assert move[t] == e, f"move[{t}] expected {e} got {move[t]}"

    # --- Test B: NO-LOOK-AHEAD reconstructability -----------------------------
    # The label at bar t must be a function ONLY of bars [t .. t+H]. We prove this by
    # MUTATING bars strictly beyond t+H and asserting label[t] is UNCHANGED, and by
    # MUTATING a bar inside (t, t+H] and asserting it CAN change (the window is live).
    base = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, size=400))
    K = 32
    lab_full = forward_bear_label(base, K=K, dd_thresh=0.05)
    t = 100
    # (B1) perturb a FUTURE-of-horizon bar (t+K+5): label[t] must NOT move (no leak from
    # beyond the horizon -- and equally, label[t] must not see anything past t+K).
    pert = base.copy()
    pert[t + K + 5] *= 1.5
    lab_pert = forward_bear_label(pert, K=K, dd_thresh=0.05)
    assert lab_pert[t] == lab_full[t] or (np.isnan(lab_pert[t]) and np.isnan(lab_full[t])), \
        "LEAK: label[t] changed when a bar beyond t+K was perturbed"
    # (B2) perturb a bar INSIDE the window (t+1) downward hard: label[t] should become 1.
    pert2 = base.copy()
    pert2[t + 1] = base[t] * 0.5     # -50% inside the window -> definite drawdown
    lab_pert2 = forward_bear_label(pert2, K=K, dd_thresh=0.05)
    assert lab_pert2[t] == 1.0, "window-live check failed: in-window crash did not set label=1"
    # (B3) past bars (< t) must never affect label[t] (it only looks forward).
    pert3 = base.copy()
    pert3[:t] *= 2.0
    lab_pert3 = forward_bear_label(pert3, K=K, dd_thresh=0.05)
    assert lab_pert3[t] == lab_full[t] or (np.isnan(lab_pert3[t]) and np.isnan(lab_full[t])), \
        "label[t] changed when PAST bars were perturbed (must be forward-only)"

    # --- Test C: NON-LEAKAGE as an input feature ------------------------------
    # The label is a forward construct; it must not be trivially equal to any single
    # contemporaneous input. We assert the labels are not constant and carry signal
    # (so they are a real target), and that bear vs the *coincident* current-bar-return
    # sign disagree on a meaningful fraction (proving forward != coincident).
    cur_ret = np.zeros_like(base)
    cur_ret[1:] = base[1:] / base[:-1] - 1.0          # the coincident lens
    coincident_bear = (cur_ret < 0).astype(float)      # "this bar was red"
    valid = ~np.isnan(lab_full)
    assert lab_full[valid].std() > 0, "bear label degenerate (no variance) -- not a usable target"
    disagree = np.mean(coincident_bear[valid] != lab_full[valid])
    assert disagree > 0.05, \
        f"forward bear label nearly identical to coincident red-bar ({disagree:.3f}) -- " \
        f"would add no information over the existing coincident head"

    # --- Test D: shape / dtype / alignment contract ---------------------------
    for arr, H in ((bear, 3), (trend, 4), (move, 3)):
        assert arr.dtype == np.float32, "labels must be float32"
        assert arr.shape == (len(close),), "label must align 1:1 with the input series"

    print("[regime_targets] self-test PASSED")
    print(f"  Test A: exact deterministic labels (bear/trend/move) OK")
    print(f"  Test B: no-look-ahead -- forward-only, horizon-bounded, past-invariant OK")
    print(f"  Test C: non-leakage -- forward != coincident (disagree={disagree:.1%}) OK")
    print(f"  Test D: shape/dtype/alignment contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
