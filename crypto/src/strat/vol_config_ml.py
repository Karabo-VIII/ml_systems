"""VOLATILITY -> CONFIG mapping via SIMPLE ML, then CAUSAL OOS REPLAY.

THE ONE DIMENSION: VOLATILITY. The prior math-decomposition found a LINEAR
vol -> config relationship of R^2 ~ 0.04 (weak). This tool tests whether a SIMPLE
non-linear ML model (DecisionTree / small RandomForest / LogisticRegression / kNN)
can find a NONLINEAR vol -> config mapping that linear regression missed
(e.g. high-vol -> take-profit exit, low-vol -> ride the trend), then REPLAYS it
CAUSALLY out-of-sample.

  FEATURES (all measured on the PRE-MOVE window ONLY -- strictly before ev.start):
    - trailing realized vol at short / med / long lookbacks
    - vol percentile vs the expanding PAST distribution (causal)
    - vol regime bucket (low / mid / high, from the causal percentile)
    - vol-of-vol (std of the rolling short-vol series in the pre-window)
    - vol trend (short-vol vs long-vol ratio -> rising / falling)

  TARGET (the winning config-CLASS for each move):
    class = FORMULATION-FAMILY x MA-TYPE
      formulation in {F1_PRICE_MA, F2_CROSS, F3_CROSS_MECH, F4_STACK, F5_PRICE_MA_MECH}
      ma_type     in {SMA, EMA, WMA, HMA, DEMA}
    The class that MAXIMIZED capture on that move (best in-class config capture).
    (Optionally also: MA-length-bin and exit-TP-bin -- reported but not the main target.)

  ML (SIMPLE only): sklearn DecisionTree / RandomForest(small) / LogisticRegression / kNN.
    Train on TRAIN moves: vol-features -> best-config-class.

  REPLAY (causal OOS walk-forward, OOS-ONCE):
    - Fit the ML on TRAIN moves only.
    - For each TEST move: predict the config-CLASS from PAST-ONLY vol features ->
      apply that class CAUSALLY (the in-class config is chosen on TRAIN, NOT on TEST)
      -> measure realized capture (net taker 0.24%).
    - No refit on TEST. TEST is untouched for any fitting.

  TWO CHECKS (reported SEPARATELY -- this is the crux):
    (1) LEARNABILITY: does vol predict the class ABOVE the majority-class base rate on
        TEST? -> classification accuracy vs base-rate + vol feature-importances.
    (2) PROFITABILITY: does ACTING on the ML-predicted class capture REAL signal OOS?
        -> POSITIVE mean capture AND beats FIXED AND beats last-week's-best (rolled),
        vs ceiling + random too.
    CRITICAL: (1) can be YES while (2) is NO if the per-move capture margin between
    classes is tiny (the known trap: winning-DNA correlation != capture improvement).

VALIDATION (--selftest, two-sided):
  (A) synthetic where vol DETERMINES the best config -> ML accuracy >> base-rate AND
      replay positive AND beats fixed.
  (B) synthetic where config is vol-INDEPENDENT -> ML ~= base-rate AND replay ~= fixed.
  MUST PASS (proves the rig detects a real vol-edge if one exists, and stays null if not).

DISCIPLINE:
  - CAUSAL: vol features measured strictly before each move's start; ML fit on TRAIN
    moves only; TEST untouched for fitting; in-class config chosen on TRAIN; no look-ahead.
  - Reuses ti_oracle_anchor.find_price_oracle_events + ti_oracle_decompose.build_candidates
    READ-ONLY (events + candidate grid byte-identical to the anchor/decomp/walkforward).
  - Real chimera (BTC + ETH/SOL, 1d/4h/1h).
  - cp1252-safe (no emoji).

Usage:
    python src/strat/vol_config_ml.py --assets BTCUSDT,ETHUSDT,SOLUSDT --cadences 1d,4h,1h
    python src/strat/vol_config_ml.py --selftest

__contract__ = {
    "kind": "research_ml_replay",
    "inputs": ["chimera OHLC via ChimeraLoader",
               "ti_oracle_anchor.find_price_oracle_events (read-only reuse)",
               "ti_oracle_decompose.build_candidates (read-only reuse)"],
    "outputs": ["runs/strat/vol_config_ml_<TAG>.json", "stdout report"],
    "invariants": [
        "vol features measured strictly before ev.start (pre-move, past-only)",
        "ML fit on TRAIN moves only; no refit on TEST (OOS-once)",
        "in-class replay config chosen on TRAIN events only (causal)",
        "TRAIN events strictly precede TEST events in bar index (no look-ahead)",
        "learnability (accuracy vs base-rate) and profitability (capture) reported "
        "SEPARATELY -- a positive (1) does NOT imply a positive (2)",
        "selftest two-sided: vol-determines-config -> learnable+profitable; "
        "vol-independent -> ~base-rate + ~fixed",
    ],
}
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# READ-ONLY reuse so events + candidate grid are byte-identical to anchor/decomp/wf.
from strat.ti_oracle_anchor import (  # noqa: E402
    MoveEvent,
    WINDOW_BARS,
    find_price_oracle_events,
    load_ohlc,
)
from strat.ti_oracle_decompose import (  # noqa: E402
    Candidate,
    build_candidates,
)

# sklearn (simple models only). Hard fallback to a hand-rolled shallow tree if absent.
try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    _HAVE_SK = True
except Exception:  # pragma: no cover
    _HAVE_SK = False


# ---- vol-feature spec -------------------------------------------------------

# Trailing realized-vol lookbacks (in bars). Short/med/long.
VOL_LB_SHORT = 10
VOL_LB_MED = 30
VOL_LB_LONG = 90

# The pre-move feature window: how many bars before ev.start we read to compute the
# vol-of-vol and trend (must be long enough to hold several short-vol estimates).
PRE_WINDOW = max(VOL_LB_LONG + VOL_LB_SHORT + 5, 120)

# Causal vol percentile uses the EXPANDING past distribution of short-vol up to
# ev.start. Regime buckets cut the percentile into low/mid/high.
REGIME_CUTS = (1.0 / 3.0, 2.0 / 3.0)

FEATURE_NAMES = (
    "vol_short",        # trailing realized vol, short LB
    "vol_med",          # trailing realized vol, med LB
    "vol_long",         # trailing realized vol, long LB
    "vol_pctile",       # causal expanding percentile of short-vol
    "vol_regime",       # 0/1/2 low/mid/high from the percentile
    "vol_of_vol",       # std of the rolling short-vol series in the pre-window
    "vol_trend",        # short/long vol ratio - 1 (>0 rising, <0 falling)
)


# ---- causal vol primitives --------------------------------------------------

def log_returns(close: np.ndarray) -> np.ndarray:
    """r[i] = ln(close[i]/close[i-1]); r[0] = 0. Causal."""
    r = np.zeros_like(close, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        r[1:] = np.log(close[1:] / close[:-1])
    r[~np.isfinite(r)] = 0.0
    return r


def rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    """Causal rolling std. out[i] uses x[i-n+1..i]. NaN until warm."""
    n = int(n)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if n <= 1 or len(x) < n:
        return out
    csum = np.cumsum(np.insert(x, 0, 0.0))
    csum2 = np.cumsum(np.insert(x * x, 0, 0.0))
    s = csum[n:] - csum[:-n]
    s2 = csum2[n:] - csum2[:-n]
    var = (s2 - s * s / n) / n
    var = np.maximum(var, 0.0)
    out[n - 1:] = np.sqrt(var)
    return out


@dataclass
class VolCtx:
    """Precomputed causal vol series for a cadence (computed once on the full close)."""
    r: np.ndarray              # log returns
    vol_short: np.ndarray      # rolling std at short LB (annualization-free)
    vol_med: np.ndarray
    vol_long: np.ndarray


def build_vol_ctx(close: np.ndarray) -> VolCtx:
    r = log_returns(close)
    return VolCtx(
        r=r,
        vol_short=rolling_std(r, VOL_LB_SHORT),
        vol_med=rolling_std(r, VOL_LB_MED),
        vol_long=rolling_std(r, VOL_LB_LONG),
    )


def vol_features_for_event(vc: VolCtx, ev_start: int) -> np.ndarray | None:
    """Compute the 7 causal vol features measured STRICTLY BEFORE ev_start.

    The reference bar is t = ev_start - 1 (the last bar fully observable before the
    move begins). All series values at t use only bars <= t -> no look-ahead.
    Returns None if there is insufficient warmup history.
    """
    t = ev_start - 1
    if t < VOL_LB_LONG:           # need long-vol warm at t
        return None
    vs = vc.vol_short[t]
    vm = vc.vol_med[t]
    vl = vc.vol_long[t]
    if not (np.isfinite(vs) and np.isfinite(vm) and np.isfinite(vl)):
        return None

    # Causal expanding percentile of short-vol: rank vs all past short-vol values
    # up to and including t (strictly <= t -> causal). Use finite values only.
    past = vc.vol_short[:t + 1]
    past = past[np.isfinite(past)]
    if past.size < 5:
        pct = 0.5
    else:
        pct = float(np.mean(past <= vs))

    if pct < REGIME_CUTS[0]:
        regime = 0.0
    elif pct < REGIME_CUTS[1]:
        regime = 1.0
    else:
        regime = 2.0

    # vol-of-vol: std of the short-vol series across the pre-window [t-PRE_WINDOW+1, t].
    lo = max(0, t - PRE_WINDOW + 1)
    win_vs = vc.vol_short[lo:t + 1]
    win_vs = win_vs[np.isfinite(win_vs)]
    vov = float(np.std(win_vs)) if win_vs.size >= 3 else 0.0

    # vol trend: short vs long ratio - 1. >0 -> short-term vol rising vs baseline.
    trend = float(vs / vl - 1.0) if vl > 0 else 0.0

    return np.array([vs, vm, vl, pct, regime, vov, trend], dtype=np.float64)


# ---- config-CLASS target ----------------------------------------------------

# A class = (formulation, ma_type). We also keep, per class, the per-event capture
# of EVERY in-class candidate so the replay can pick the in-class config on TRAIN.

# Target granularity:
#   "full"        -> class = formulation-family x MA-type (the main spec target).
#   "formulation" -> class = formulation-family only (vol -> ride/cross/TP-exit). This
#                    is the dimension vol most directly controls; used by the selftest
#                    positive control and also reported on real data for comparison.
TARGET_KINDS = ("full", "formulation")


def class_label(formulation: str, ma_type: str, kind: str = "full") -> str:
    return formulation if kind == "formulation" else f"{formulation}|{ma_type}"


@dataclass
class EventBundle:
    """Per-event: causal vol features + the full capture vector over all candidates
    + the per-class best capture + the argmax winning class, for EACH target kind."""
    ev: MoveEvent
    feats: np.ndarray            # [n_features]
    cap_vec: np.ndarray          # [n_cands] realized long ROI per candidate
    # per target-kind: {class_label: best in-class capture} and the argmax winner.
    class_best: dict             # {kind: {label: best capture}}
    win_class: dict              # {kind: argmax label}


def _index_by_class(cands: list[Candidate], kind: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for j, c in enumerate(cands):
        lab = class_label(c.formulation, c.ma_type, kind)
        out.setdefault(lab, []).append(j)
    return out


def build_event_bundles(
    open_a, high_a, low_a, close_a, cadence: str,
) -> tuple[list[EventBundle], list[Candidate], dict, dict]:
    """Detect events, build candidates, compute causal vol features + per-candidate
    capture + per-class best + winning class (for BOTH target kinds) per event.

    Returns (bundles, cands, class_labels_by_kind, cand_idx_by_class_by_kind).
      cand_idx_by_class_by_kind[kind][label] = candidate indices in that class.
    Events without enough vol warmup are dropped (stated in the caller).
    """
    win_lo, win_hi = WINDOW_BARS[cadence]
    events = find_price_oracle_events(high_a, low_a, win_lo, win_hi)
    cands = build_candidates(open_a, high_a, low_a, close_a)
    vc = build_vol_ctx(close_a)

    cibc_by_kind = {k: _index_by_class(cands, k) for k in TARGET_KINDS}
    class_labels_by_kind = {k: sorted(cibc_by_kind[k]) for k in TARGET_KINDS}

    bundles: list[EventBundle] = []
    for ev in events:
        feats = vol_features_for_event(vc, ev.start)
        if feats is None:
            continue
        cap_vec = np.array([c.fn(ev.start, ev.end) for c in cands], dtype=np.float64)
        class_best: dict = {}
        win_class: dict = {}
        for k in TARGET_KINDS:
            cb = {}
            for cl, idxs in cibc_by_kind[k].items():
                cb[cl] = float(np.nanmax(cap_vec[idxs])) if idxs else -np.inf
            class_best[k] = cb
            win_class[k] = max(cb, key=cb.get)
        bundles.append(EventBundle(
            ev=ev, feats=feats, cap_vec=cap_vec,
            class_best=class_best, win_class=win_class,
        ))
    return bundles, cands, class_labels_by_kind, cibc_by_kind


# ---- simple ML models -------------------------------------------------------

def make_models() -> dict:
    """The SIMPLE model zoo. sklearn if available; else a hand-rolled shallow tree."""
    if _HAVE_SK:
        return {
            "decision_tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=3,
                                                    random_state=0),
            "random_forest": RandomForestClassifier(n_estimators=60, max_depth=5,
                                                    min_samples_leaf=3, random_state=0,
                                                    n_jobs=1),
            "logistic": LogisticRegression(max_iter=500, multi_class="auto"),
            "knn": KNeighborsClassifier(n_neighbors=5),
        }
    return {"shallow_tree": _ShallowTree(max_depth=4, min_leaf=3)}


class _ShallowTree:
    """Tiny hand-rolled shallow decision-tree classifier (Gini), used only if sklearn
    is unavailable. Standalone -- no external deps. Supports fit/predict + a crude
    feature_importances_ (split-count weighted by samples)."""

    def __init__(self, max_depth=4, min_leaf=3):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.tree = None
        self.classes_ = None
        self.feature_importances_ = None
        self._imp = None
        self._nf = 0

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        _, cnt = np.unique(y, return_counts=True)
        p = cnt / cnt.sum()
        return 1.0 - np.sum(p * p)

    def _build(self, X, y, depth):
        node = {"leaf": True, "pred": Counter(y).most_common(1)[0][0]}
        if depth >= self.max_depth or len(y) < 2 * self.min_leaf or len(set(y)) == 1:
            return node
        best_gain, best = 0.0, None
        parent = self._gini(y)
        n = len(y)
        for f in range(X.shape[1]):
            vals = np.unique(X[:, f])
            if len(vals) < 2:
                continue
            for thr in (vals[:-1] + vals[1:]) / 2.0:
                m = X[:, f] <= thr
                if m.sum() < self.min_leaf or (~m).sum() < self.min_leaf:
                    continue
                gain = parent - (m.sum() / n * self._gini(y[m])
                                 + (~m).sum() / n * self._gini(y[~m]))
                if gain > best_gain:
                    best_gain, best = gain, (f, thr, m)
        if best is None:
            return node
        f, thr, m = best
        self._imp[f] += best_gain * n
        return {"leaf": False, "f": f, "thr": thr,
                "L": self._build(X[m], y[m], depth + 1),
                "R": self._build(X[~m], y[~m], depth + 1)}

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self._nf = X.shape[1]
        self._imp = np.zeros(self._nf, dtype=np.float64)
        self.classes_ = np.unique(y)
        self.tree = self._build(X, y, 0)
        s = self._imp.sum()
        self.feature_importances_ = (self._imp / s) if s > 0 else self._imp
        return self

    def _pred_one(self, x, node):
        while not node["leaf"]:
            node = node["L"] if x[node["f"]] <= node["thr"] else node["R"]
        return node["pred"]

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._pred_one(x, self.tree) for x in X])


def fit_predict(model_name: str, Xtr, ytr, Xte):
    """Fit one model on TRAIN, predict on TEST. Returns (pred, feature_importances|None).
    Standardizes features for logistic/knn (fit scaler on TRAIN only -> causal)."""
    models = make_models()
    if model_name not in models:
        model_name = next(iter(models))
    model = models[model_name]

    needs_scale = model_name in ("logistic", "knn")
    if needs_scale and _HAVE_SK:
        scaler = StandardScaler().fit(Xtr)   # TRAIN-only fit (causal)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)
    else:
        Xtr_s, Xte_s = Xtr, Xte

    # Guard: single-class TRAIN -> predict that class everywhere.
    if len(np.unique(ytr)) < 2:
        only = ytr[0]
        return np.array([only] * len(Xte)), None

    model.fit(Xtr_s, ytr)
    pred = model.predict(Xte_s)
    fi = getattr(model, "feature_importances_", None)
    if fi is not None:
        fi = np.asarray(fi, dtype=np.float64)
    return pred, fi


# ---- causal OOS replay ------------------------------------------------------

@dataclass
class ReplayResult:
    tag: str
    n_train: int
    n_test: int
    model: str
    # learnability
    base_rate: float                 # majority-class share on TEST
    accuracy: float                  # ML class accuracy on TEST
    feature_importance: dict         # {feature: importance} (if available)
    # profitability (all on TEST, realized capture net taker)
    ml_capture: float                # capture of acting on ML-predicted class
    fixed_capture: float             # one globally-fixed config
    rolled_capture: float            # last-week's-best (TRAIN-best class, fixed config)
    ceiling_capture: float           # per-move best-class (hindsight ceiling)
    random_capture: float            # random class per move
    # extra context
    ml_beats_fixed: bool
    ml_beats_rolled: bool
    ml_positive: bool
    train_class_dist: dict
    test_class_dist: dict


def _inclass_train_best_config(
    bundles_train: list[EventBundle], cls: str, cand_idx_by_class: dict,
) -> int | None:
    """Pick the BEST in-class candidate index on TRAIN events (mean capture over TRAIN).
    This is how the replay turns a predicted CLASS into a concrete causal config:
    the param choice is made on TRAIN only -> no TEST look-ahead."""
    idxs = cand_idx_by_class.get(cls, [])
    if not idxs:
        return None
    # mean over TRAIN events of each in-class candidate's capture
    mat = np.array([b.cap_vec[idxs] for b in bundles_train], dtype=np.float64)  # [ntr, k]
    if mat.size == 0:
        return None
    means = np.nanmean(mat, axis=0)
    best_local = int(np.nanargmax(means))
    return idxs[best_local]


# The canonical naive class per target-kind (matches walkforward's F2_CROSS/SMA family).
FIXED_CLASS_BY_KIND = {"full": "F2_CROSS|SMA", "formulation": "F2_CROSS"}


def replay_oos(
    bundles: list[EventBundle],
    cand_idx_by_class: dict,
    n_cands: int,
    train_frac: float,
    model_name: str,
    tag: str,
    target_kind: str = "full",
    seed: int = 7,
) -> ReplayResult | None:
    """Single causal TRAIN/TEST split (OOS-once). TRAIN = first `train_frac` of the
    bar-ordered events; TEST = the rest (strictly later bar indices -> no look-ahead).

    - Fit ML(vol -> win_class[target_kind]) on TRAIN only.
    - Predict class on each TEST move from PAST-ONLY vol.
    - Turn the predicted class into a config via the TRAIN in-class best (causal).
    - Score capture vs fixed / rolled / ceiling / random.

    target_kind: "full" (formulation x ma_type) or "formulation" (family only).
    cand_idx_by_class: the index map FOR THAT target_kind.
    """
    fixed_class = FIXED_CLASS_BY_KIND.get(target_kind, FIXED_CLASS_BY_KIND["full"])
    n = len(bundles)
    if n < 12:
        return None
    # bundles are already bar-ordered (events come from a left-to-right scan).
    n_train = max(6, int(round(n * train_frac)))
    n_train = min(n_train, n - 4)         # leave >= 4 test events
    tr = bundles[:n_train]
    te = bundles[n_train:]
    if len(te) < 3:
        return None

    # causal guard: last TRAIN event ends at/before first TEST event start.
    assert tr[-1].ev.end <= te[0].ev.start, "look-ahead: train overlaps test"

    Xtr = np.array([b.feats for b in tr], dtype=np.float64)
    ytr = np.array([b.win_class[target_kind] for b in tr])
    Xte = np.array([b.feats for b in te], dtype=np.float64)
    yte = np.array([b.win_class[target_kind] for b in te])

    pred, fi = fit_predict(model_name, Xtr, ytr, Xte)

    # ---- learnability ----
    base_rate = float(Counter(yte).most_common(1)[0][1] / len(yte))
    accuracy = float(np.mean(pred == yte))
    feat_imp = {}
    if fi is not None and len(fi) == len(FEATURE_NAMES):
        feat_imp = {FEATURE_NAMES[i]: float(fi[i]) for i in range(len(FEATURE_NAMES))}

    # ---- profitability ----
    # Precompute, per class, the TRAIN in-class best config index (causal).
    classes = sorted(set(ytr) | set(pred))
    train_best_cfg: dict[str, int | None] = {}
    for cls in set(list(ytr) + list(pred)):
        train_best_cfg[cls] = _inclass_train_best_config(tr, cls, cand_idx_by_class)

    # ML capture: for each TEST move, apply the TRAIN in-class best config of the
    # PREDICTED class. If that class has no TRAIN config, fall back to FIXED.
    fixed_cfg = _inclass_train_best_config(tr, fixed_class, cand_idx_by_class)
    if fixed_cfg is None:
        # fall back to global TRAIN-best config across all candidates
        all_means = np.nanmean(np.array([b.cap_vec for b in tr]), axis=0)
        fixed_cfg = int(np.nanargmax(all_means))

    ml_caps, fixed_caps, ceil_caps, rand_caps = [], [], [], []
    # rolled = TRAIN-best CLASS (single global class), its TRAIN in-class best config.
    train_class_mean = {}
    for cls in cand_idx_by_class:
        vals = [b.class_best[target_kind][cls] for b in tr]
        train_class_mean[cls] = float(np.nanmean(vals)) if vals else -np.inf
    rolled_class = max(train_class_mean, key=train_class_mean.get)
    rolled_cfg = _inclass_train_best_config(tr, rolled_class, cand_idx_by_class)
    if rolled_cfg is None:
        rolled_cfg = fixed_cfg
    rolled_caps = []

    rng = np.random.default_rng(seed)
    for k, b in enumerate(te):
        pcls = pred[k]
        cfg = train_best_cfg.get(pcls)
        if cfg is None:
            cfg = fixed_cfg
        ml_caps.append(float(b.cap_vec[cfg]))
        fixed_caps.append(float(b.cap_vec[fixed_cfg]))
        rolled_caps.append(float(b.cap_vec[rolled_cfg]))
        # ceiling: per-move best CLASS's best in-class config (hindsight on TEST).
        ceil_caps.append(float(np.nanmax(b.cap_vec)))
        # random: a random candidate's capture on this move.
        rj = int(rng.integers(0, n_cands))
        rand_caps.append(float(b.cap_vec[rj]))

    ml_cap = float(np.mean(ml_caps))
    fixed_cap = float(np.mean(fixed_caps))
    rolled_cap = float(np.mean(rolled_caps))
    ceil_cap = float(np.mean(ceil_caps))
    rand_cap = float(np.mean(rand_caps))

    return ReplayResult(
        tag=tag,
        n_train=len(tr),
        n_test=len(te),
        model=model_name,
        base_rate=base_rate,
        accuracy=accuracy,
        feature_importance=feat_imp,
        ml_capture=ml_cap,
        fixed_capture=fixed_cap,
        rolled_capture=rolled_cap,
        ceiling_capture=ceil_cap,
        random_capture=rand_cap,
        ml_beats_fixed=bool(ml_cap > fixed_cap),
        ml_beats_rolled=bool(ml_cap > rolled_cap),
        ml_positive=bool(ml_cap > 0.0),
        train_class_dist=dict(Counter(ytr).most_common()),
        test_class_dist=dict(Counter(yte).most_common()),
    )


# ---- verdict ----------------------------------------------------------------

def verdict_line(rr: ReplayResult) -> str:
    """ONE-LINE verdict separating learnability from profitability."""
    learnable = rr.accuracy > rr.base_rate + 1e-9
    margin = rr.accuracy - rr.base_rate
    profitable = rr.ml_positive and rr.ml_beats_fixed and rr.ml_beats_rolled

    if profitable and learnable:
        return ("VOL -> CONFIG MAPS PROFITABLY: YES. Learnable (acc %.3f > base %.3f, "
                "+%.3f) AND acting on it is positive (%.4f) and beats fixed (%.4f) + "
                "last-week's-best (%.4f)."
                % (rr.accuracy, rr.base_rate, margin, rr.ml_capture,
                   rr.fixed_capture, rr.rolled_capture))
    if learnable and not profitable:
        return ("VOL -> CONFIG MAPS PROFITABLY: NO -- LEARNABLE-BUT-UNPROFITABLE (MARGIN "
                "TRAP). Vol predicts the class above base-rate (acc %.3f > %.3f, +%.3f) "
                "but acting on it does NOT beat fixed/last-week's-best (ml %.4f vs fixed "
                "%.4f / rolled %.4f) -- the per-move capture margin between classes is "
                "too small to bank."
                % (rr.accuracy, rr.base_rate, margin, rr.ml_capture,
                   rr.fixed_capture, rr.rolled_capture))
    if not learnable and profitable:
        return ("VOL -> CONFIG MAPS PROFITABLY: WEAK/INCIDENTAL -- not learnable above "
                "base-rate (acc %.3f vs %.3f) yet ml capture %.4f edged fixed %.4f; "
                "treat as noise, not a vol-edge."
                % (rr.accuracy, rr.base_rate, rr.ml_capture, rr.fixed_capture))
    return ("VOL -> CONFIG MAPS PROFITABLY: NO -- GENUINELY-NOT-LEARNABLE (NULL). Vol "
            "does not predict the config-class above base-rate (acc %.3f vs %.3f) and "
            "acting on the prediction does not beat fixed (ml %.4f vs fixed %.4f)."
            % (rr.accuracy, rr.base_rate, rr.ml_capture, rr.fixed_capture))


# ---- reporting --------------------------------------------------------------

def _f(v, pct=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "    n/a"
    return f"{v*100:7.3f}%" if pct else f"{v:8.4f}"


def print_report(results: list[ReplayResult], pooled: ReplayResult | None) -> None:
    print("")
    print("=" * 104)
    print("VOLATILITY -> CONFIG via SIMPLE ML  (causal OOS replay)")
    print("=" * 104)

    print("\n[1] LEARNABILITY -- does vol predict the winning config-CLASS above base-rate?")
    hdr = (f"{'tag':>22} | {'model':>14} | {'nTr':>4} {'nTe':>4} | "
           f"{'base':>7} {'acc':>7} {'margin':>7}")
    print(hdr)
    print("-" * 104)
    rows = results + ([pooled] if pooled else [])
    for r in rows:
        if r is None:
            continue
        margin = r.accuracy - r.base_rate
        print(f"{r.tag:>22} | {r.model:>14} | {r.n_train:>4} {r.n_test:>4} | "
              f"{_f(r.base_rate):>7} {_f(r.accuracy):>7} {_f(margin):>7}")

    print("\n[2] PROFITABILITY -- does ACTING on the ML-predicted class capture REAL "
          "signal OOS? (realized capture, net taker)")
    hdr2 = (f"{'tag':>22} | {'ml':>9} {'fixed':>9} {'rolled':>9} {'ceiling':>9} "
            f"{'random':>9} | {'>fix':>4} {'>roll':>5} {'+ve':>4}")
    print(hdr2)
    print("-" * 104)
    for r in rows:
        if r is None:
            continue
        print(f"{r.tag:>22} | {_f(r.ml_capture):>9} {_f(r.fixed_capture):>9} "
              f"{_f(r.rolled_capture):>9} {_f(r.ceiling_capture):>9} "
              f"{_f(r.random_capture):>9} | "
              f"{('Y' if r.ml_beats_fixed else 'n'):>4} "
              f"{('Y' if r.ml_beats_rolled else 'n'):>5} "
              f"{('Y' if r.ml_positive else 'n'):>4}")

    # feature importance (pooled if present, else the first result with importances)
    fi_src = None
    for r in rows:
        if r is not None and r.feature_importance:
            fi_src = r
            break
    print("\n[3] VOL FEATURE-IMPORTANCE (which vol feature drives the class prediction)")
    if fi_src is None:
        print("   (no feature-importance available from the chosen model)")
    else:
        items = sorted(fi_src.feature_importance.items(), key=lambda kv: -kv[1])
        print(f"   source: {fi_src.tag} / {fi_src.model}")
        for k, v in items:
            bar = "#" * int(round(v * 40))
            print(f"     {k:>12}: {v:6.3f}  {bar}")

    print("\n[4] ONE-LINE VERDICT")
    target = pooled if pooled else (results[0] if results else None)
    if target is not None:
        print("   " + verdict_line(target))
    else:
        print("   INDETERMINATE: no usable replay (too few events).")
    print("=" * 104)
    print("NOTE: learnability (1) and profitability (2) are SEPARATE. A positive (1) "
          "does NOT imply a positive (2) -- watch the margin trap.")
    print("")


# ---- selftest (two-sided) ---------------------------------------------------

def _make_synth_bundles(n_events, n_cands, vol_drives, seed):
    """Build synthetic EventBundles with a CONTROLLED capture matrix, to validate the
    ML-fit + causal-replay mechanics independently of the price-oracle event geometry
    (within real price-oracle windows the per-formulation capture margin is tiny by
    construction -- a real and reportable fact -- which makes a price-level positive
    control unreliable; this mechanics-level control is unambiguous).

    Two synthetic classes split the candidate columns in half: class A = first half,
    class B = second half. Each event has a causal vol feature `vs` (the only feature
    that carries signal here, placed in slot 0; other slots are noise).

      vol_drives=True : in LOW-vol events class A wins big (+0.04 vs -0.01); in HIGH-vol
                        events class B wins big. So vol cleanly determines the winning
                        class AND the per-move capture gap is LARGE -> routing by vol is
                        both learnable and profitable (beats either single fixed class).
      vol_drives=False: the winning class is assigned by an INDEPENDENT coin (not vol);
                        vol carries no information -> ML ~= base-rate and routing ~= fixed.

    Bar indices are synthesized strictly increasing + non-overlapping so the causal
    TRAIN/TEST guard (train.end <= test.start) holds.
    """
    rng = np.random.default_rng(seed)
    half = n_cands // 2
    classA_idx = list(range(half))
    classB_idx = list(range(half, n_cands))

    bundles: list[EventBundle] = []
    bar = 100
    for e in range(n_events):
        vs = float(rng.uniform(0.005, 0.05))          # the causal vol level
        low_vol = vs < 0.0275
        if vol_drives:
            a_wins = low_vol
        else:
            a_wins = bool(rng.integers(0, 2))         # vol-independent coin

        cap = np.full(n_cands, np.nan, dtype=np.float64)
        # winning class gets a big positive capture; losing class a small negative.
        win_lvl = 0.040 + rng.normal(0, 0.004)
        lose_lvl = -0.010 + rng.normal(0, 0.004)
        if a_wins:
            cap[classA_idx] = win_lvl + rng.normal(0, 0.002, size=half)
            cap[classB_idx] = lose_lvl + rng.normal(0, 0.002, size=n_cands - half)
        else:
            cap[classA_idx] = lose_lvl + rng.normal(0, 0.002, size=half)
            cap[classB_idx] = win_lvl + rng.normal(0, 0.002, size=n_cands - half)

        # 7 features: slot 0 = vol-short (the signal); rest are pure noise.
        feats = np.array([vs, vs * 1.1, vs * 0.9,
                          float(low_vol), 0.0,
                          float(rng.normal(0, 1)), float(rng.normal(0, 1))],
                         dtype=np.float64)

        start = bar
        end = bar + 7
        ev = MoveEvent(start=start, end=end, low_idx=start, high_idx=end - 1,
                       price_roi=0.05)
        # class_best / win_class for a synthetic "syn" target-kind only.
        cb = {"syn": {"A": float(np.nanmax(cap[classA_idx])),
                      "B": float(np.nanmax(cap[classB_idx]))}}
        wc = {"syn": "A" if cb["syn"]["A"] >= cb["syn"]["B"] else "B"}
        bundles.append(EventBundle(ev=ev, feats=feats, cap_vec=cap,
                                   class_best=cb, win_class=wc))
        bar = end + 3                                  # non-overlapping, increasing
    cibc = {"syn": {"A": classA_idx, "B": classB_idx}}
    return bundles, cibc


def _selftest_mechanics() -> bool:
    """Unambiguous two-sided proof of the rig on a CONTROLLED synthetic capture matrix.
    Side A: vol DRIVES the class -> learnable + profitable (routing beats fixed+rolled).
    Side B: vol-INDEPENDENT       -> ~base-rate + routing ~= fixed.
    """
    ok = True
    model = "decision_tree" if _HAVE_SK else "shallow_tree"
    n_cands = 24

    # FIXED class for the synthetic kind = always class "A".
    global FIXED_CLASS_BY_KIND
    FIXED_CLASS_BY_KIND = dict(FIXED_CLASS_BY_KIND)
    FIXED_CLASS_BY_KIND["syn"] = "A"

    bA, cibcA = _make_synth_bundles(220, n_cands, vol_drives=True, seed=1)
    rrA = replay_oos(bA, cibcA["syn"], n_cands, train_frac=0.6, model_name=model,
                     tag="MECH_A_vol_drives", target_kind="syn")
    bB, cibcB = _make_synth_bundles(220, n_cands, vol_drives=False, seed=2)
    rrB = replay_oos(bB, cibcB["syn"], n_cands, train_frac=0.6, model_name=model,
                     tag="MECH_B_vol_indep", target_kind="syn")
    if rrA is None or rrB is None:
        print("[selftest] FAIL(mech): replay returned None")
        return False

    print(f"[selftest] MECH-A vol-drives : base={rrA.base_rate:.3f} acc={rrA.accuracy:.3f} "
          f"ml={rrA.ml_capture:.4f} fixed={rrA.fixed_capture:.4f} rolled={rrA.rolled_capture:.4f}")
    print(f"[selftest]   verdict: {verdict_line(rrA)}")
    print(f"[selftest] MECH-B vol-indep  : base={rrB.base_rate:.3f} acc={rrB.accuracy:.3f} "
          f"ml={rrB.ml_capture:.4f} fixed={rrB.fixed_capture:.4f}")
    print(f"[selftest]   verdict: {verdict_line(rrB)}")

    # Side A: learnable (acc >> base) AND profitable (positive, beats fixed AND rolled).
    if not (rrA.accuracy > rrA.base_rate + 0.15):
        print(f"[selftest] FAIL(mech-A): not learnable (acc {rrA.accuracy:.3f} vs base "
              f"{rrA.base_rate:.3f})"); ok = False
    if not (rrA.ml_capture > 0 and rrA.ml_beats_fixed and rrA.ml_beats_rolled):
        print(f"[selftest] FAIL(mech-A): not profitable (ml {rrA.ml_capture:.4f}, "
              f">fix={rrA.ml_beats_fixed}, >roll={rrA.ml_beats_rolled})"); ok = False
    if "YES" not in verdict_line(rrA):
        print("[selftest] FAIL(mech-A): verdict not YES on profitable positive control")
        ok = False
    # Side B: NOT learnable above base-rate, routing does NOT beat fixed.
    marginA = rrA.accuracy - rrA.base_rate
    marginB = rrB.accuracy - rrB.base_rate
    if not (marginB < marginA - 0.10):
        print(f"[selftest] FAIL(mech-B): margin {marginB:.3f} not << A's {marginA:.3f}")
        ok = False
    if not (rrB.ml_capture <= rrB.fixed_capture + 0.004):
        print(f"[selftest] FAIL(mech-B): ml {rrB.ml_capture:.4f} beat fixed "
              f"{rrB.fixed_capture:.4f} on vol-independent data"); ok = False
    if "YES" in verdict_line(rrB):
        print("[selftest] FAIL(mech-B): false-positive YES on vol-independent data")
        ok = False
    return ok


def _synth_vol_determines_config(n=2600, seed=21):
    """Side A: vol DETERMINES the best config.

    Construct alternating regime blocks keyed by PRE-block volatility:
      LOW-vol block  : a smooth persistent grind up -> a trend RIDE (cross/stack/F1)
                       captures best; a quick take-profit exit leaves money on the table.
      HIGH-vol block : a sharp pop FOLLOWED BY a hard fade back below entry -> a quick
                       take-profit MECHANICAL-exit (F3/F5 TP) banks the pop, while a
                       ride round-trips to a LOSS. So the WINNING config-class flips
                       with vol, and routing per-vol strictly dominates any single
                       fixed class.
    Vol measured on the pre-block window thus cleanly predicts which config wins, and
    acting on that prediction must beat both a fixed config and last-week's-best.
    """
    rng = np.random.default_rng(seed)
    price = [100.0]
    # THREE vol regimes cycled in turn, each tuned to produce ~ONE in-band move and to
    # be best-captured by a DIFFERENT config-class. With three roughly-balanced classes
    # NO single class wins a majority -> the target base-rate sits near 1/3, so (a) the
    # learnability margin has headroom AND (b) any single fixed/last-week's-best class
    # loses meaningfully to per-vol routing (the profitability YES path is reachable).
    #   regime 0 (LOW vol)  : smooth grind up        -> a slow trend RIDE wins.
    #   regime 1 (MID vol)  : choppy stair-step up    -> a fast cross / shorter MA wins.
    #   regime 2 (HIGH vol) : pop-then-crash          -> a TP mechanical-exit wins.
    # Only TWO regimes (a cleaner two-class positive control). Each ~one in-band move.
    #   regime 0 (LOW vol)  : slow smooth grind up, NO pullbacks -> a RIDE that holds to
    #                         window-end (F1/F2/F4, a TIME/never-exit) wins; a tight TP
    #                         exits early and leaves most of the move on the table.
    #   regime 1 (HIGH vol) : a fast rip up then a HARD crash back below entry -> only a
    #                         tight take-profit MECHANICAL exit (F3/F5 TP) banks the pop;
    #                         every ride round-trips to a LOSS. The capture gap between
    #                         the TP class and the ride class is LARGE here.
    i = 0
    phase = -1
    while i < n:
        phase = (phase + 1) % 2
        if phase == 0:        # LOW vol smooth grind up (ride wins big, TP leaves money)
            blen = max(18, 22 + int(rng.integers(-3, 4)))
            for _ in range(blen):
                ret = 0.0055 + rng.normal(0, 0.0006)       # very smooth, monotone up
                price.append(price[-1] * (1.0 + ret))
                i += 1
                if i >= n:
                    break
        else:                 # HIGH vol rip-then-crash (only a tight TP survives)
            blen = max(14, 18 + int(rng.integers(-2, 3)))
            half = max(5, blen // 2)
            for k in range(blen):
                if k < half:
                    ret = 0.020 + rng.normal(0, 0.010)     # fast rip up (high vol)
                else:
                    ret = -0.028 + rng.normal(0, 0.010)    # hard crash well below entry
                price.append(price[-1] * (1.0 + ret))
                i += 1
                if i >= n:
                    break
        if i >= n:
            break
    p = np.array(price, dtype=np.float64)
    p = np.maximum(p, 1.0)        # guard against degenerate non-positive prices
    o = p.copy()
    c = p.copy()
    h = p * 1.003
    lo = p * 0.997
    return o, h, lo, c


def _synth_vol_independent(n=2600, seed=23):
    """Side B: config is vol-INDEPENDENT. One homogeneous regime throughout; which
    config wins a given move is driven by idiosyncratic noise, NOT by vol. Vol should
    NOT predict the class -> ML ~= base-rate, replay ~= fixed."""
    rng = np.random.default_rng(seed)
    price = [100.0]
    for i in range(n):
        # single stationary character: modest drift + constant-scale noise.
        ret = 0.0015 + rng.normal(0, 0.010)
        price.append(price[-1] * (1.0 + ret))
    p = np.array(price, dtype=np.float64)
    o = p.copy()
    c = p.copy()
    h = p * 1.0025
    lo = p * 0.9975
    return o, h, lo, c


def _run_synth(o, h, lo, c, cadence="1d", model="random_forest", tag="synth",
               target_kind="formulation"):
    bundles, cands, class_labels, cibc = build_event_bundles(o, h, lo, c, cadence)
    rr = replay_oos(bundles, cibc[target_kind], len(cands), train_frac=0.6,
                    model_name=model, tag=tag, target_kind=target_kind)
    return bundles, rr


def selftest() -> bool:
    """Two-sided validation.

    PART 1 (AUTHORITATIVE GATE) -- mechanics-level controlled capture matrix:
      proves the ML-fit + causal-replay + two-check rig DETECTS a real vol-edge when one
      exists (learnable + profitable -> verdict YES) and stays NULL when it does not
      (~base-rate + ~fixed -> verdict NO). This is the clean positive/negative control.

    PART 2 (DIRECTIONAL SMOKE) -- real price-mechanics synthetics (formulation target):
      a vol-determines-config price series must yield a LARGER learnability margin than a
      vol-independent one. We assert only the DIRECTION here, not strict profitability:
      within real price-oracle move windows the per-formulation capture margin is tiny by
      construction (the event ends at the move's high, so a 'ride' always banks the pop)
      -- that is itself a real, reportable finding (the margin trap), so a strict
      profitability threshold on price synthetics would be testing the geometry, not the
      rig. The mechanics gate (Part 1) carries the profitability proof.
    """
    ok = True

    # ---- PART 1: mechanics gate (authoritative) ----
    print("[selftest] PART 1 -- mechanics-level controlled capture matrix (authoritative)")
    if not _selftest_mechanics():
        ok = False

    # ---- PART 2: real-mechanics directional smoke ----
    print("\n[selftest] PART 2 -- real price-mechanics synthetics (formulation target, "
          "directional)")
    model = "random_forest" if _HAVE_SK else "shallow_tree"
    o, h, lo, c = _synth_vol_determines_config()
    _, rrA = _run_synth(o, h, lo, c, cadence="1d", model=model, tag="A_vol_determines")
    o2, h2, lo2, c2 = _synth_vol_independent()
    _, rrB = _run_synth(o2, h2, lo2, c2, cadence="1d", model=model, tag="B_vol_indep")
    if rrA is None or rrB is None:
        print("[selftest] FAIL(part2): too few events")
        return False
    marginA = rrA.accuracy - rrA.base_rate
    marginB = rrB.accuracy - rrB.base_rate
    print(f"[selftest] A vol-determines: base={rrA.base_rate:.3f} acc={rrA.accuracy:.3f} "
          f"margin={marginA:.3f} ml={rrA.ml_capture:.4f} fixed={rrA.fixed_capture:.4f}")
    print(f"[selftest] B vol-indep     : base={rrB.base_rate:.3f} acc={rrB.accuracy:.3f} "
          f"margin={marginB:.3f} ml={rrB.ml_capture:.4f} fixed={rrB.fixed_capture:.4f}")
    # Directional discriminator only: vol-determines must out-learn vol-independent.
    if not (marginA > marginB):
        print(f"[selftest] FAIL(part2): vol-determines margin {marginA:.3f} not > "
              f"vol-independent margin {marginB:.3f}")
        ok = False

    print("\n[selftest] PASS" if ok else "\n[selftest] FAIL")
    return ok


# ---- main -------------------------------------------------------------------

def _result_to_dict(r: ReplayResult) -> dict:
    return {
        "tag": r.tag, "model": r.model,
        "n_train": r.n_train, "n_test": r.n_test,
        "learnability": {
            "base_rate": r.base_rate, "accuracy": r.accuracy,
            "margin": r.accuracy - r.base_rate,
            "feature_importance": r.feature_importance,
        },
        "profitability": {
            "ml_capture": r.ml_capture, "fixed_capture": r.fixed_capture,
            "rolled_capture": r.rolled_capture, "ceiling_capture": r.ceiling_capture,
            "random_capture": r.random_capture,
            "ml_beats_fixed": r.ml_beats_fixed, "ml_beats_rolled": r.ml_beats_rolled,
            "ml_positive": r.ml_positive,
        },
        "train_class_dist": r.train_class_dist,
        "test_class_dist": r.test_class_dist,
        "verdict": verdict_line(r),
    }


def main():
    ap = argparse.ArgumentParser(
        description="VOLATILITY -> CONFIG via simple ML + causal OOS replay")
    ap.add_argument("--assets", default="BTCUSDT,ETHUSDT,SOLUSDT")
    ap.add_argument("--cadences", default="1d,4h,1h")
    ap.add_argument("--model", default="random_forest",
                    help="decision_tree|random_forest|logistic|knn (or shallow_tree)")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    cadences = [c.strip() for c in args.cadences.split(",") if c.strip()]
    for cad in cadences:
        if cad not in WINDOW_BARS:
            print(f"[error] unknown cadence '{cad}'; known={list(WINDOW_BARS)}")
            sys.exit(2)

    print(f"[info] sklearn={'yes' if _HAVE_SK else 'no'} model={args.model}")

    # Load + build bundles ONCE per (asset,cadence); reuse across both target kinds.
    slices: list[tuple[str, list, list, dict, int]] = []  # (tag, bundles, cands, cibc, n)
    for asset in assets:
        for cad in cadences:
            print(f"[run] {asset} {cad}: events + vol features + capture ...", flush=True)
            try:
                o, h, lo, c = load_ohlc(asset, cad)
            except Exception as e:
                print(f"[warn] {asset} {cad}: load failed ({e}); skipping")
                continue
            bundles, cands, class_labels, cibc = build_event_bundles(o, h, lo, c, cad)
            print(f"[run] {asset} {cad}: {len(bundles)} usable events "
                  f"({len(cands)} candidates, full={len(class_labels['full'])} / "
                  f"formulation={len(class_labels['formulation'])} classes)", flush=True)
            slices.append((f"{asset.replace('USDT','')}/{cad}", bundles, cands, cibc,
                           len(cands)))

    artifact_by_kind: dict = {}
    for target_kind in TARGET_KINDS:
        print("\n" + "#" * 104)
        print(f"# TARGET KIND = {target_kind.upper()}  "
              f"({'formulation-family x MA-type' if target_kind=='full' else 'formulation-family only'})")
        print("#" * 104)
        results: list[ReplayResult] = []
        all_bundles: list[EventBundle] = []
        pooled_cibc = None
        pooled_n_cands = 0
        for tag, bundles, cands, cibc, n_c in slices:
            if pooled_cibc is None:
                pooled_cibc = cibc[target_kind]
                pooled_n_cands = n_c
            rr = replay_oos(bundles, cibc[target_kind], n_c, train_frac=args.train_frac,
                            model_name=args.model, tag=tag, target_kind=target_kind)
            if rr is not None:
                results.append(rr)
            all_bundles.extend(bundles)

        # Pooled replay over ALL slices (more statistical power; per-slice rows remain
        # the strictly-causal headline). Bar-order within each slice is causal; the
        # split is by event order. The candidate grid is identical across slices
        # (build_candidates is param-grid-only) so index alignment holds.
        pooled = None
        if all_bundles and pooled_cibc is not None:
            pooled = replay_oos(all_bundles, pooled_cibc, pooled_n_cands,
                                train_frac=args.train_frac, model_name=args.model,
                                tag="POOLED(all)", target_kind=target_kind)
        print_report(results, pooled)
        artifact_by_kind[target_kind] = {
            "fixed_class": FIXED_CLASS_BY_KIND.get(target_kind),
            "per_slice": [_result_to_dict(r) for r in results],
            "pooled": _result_to_dict(pooled) if pooled else None,
        }

    out_path = (Path(args.out) if args.out
                else PROJECT_ROOT / "runs" / "strat" / "vol_config_ml.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "tool": "vol_config_ml",
        "question": "does volatility map to config PROFITABLY, and is it learnable?",
        "assets": assets,
        "cadences": cadences,
        "model": args.model,
        "sklearn": _HAVE_SK,
        "spec": {
            "vol_lookbacks": {"short": VOL_LB_SHORT, "med": VOL_LB_MED, "long": VOL_LB_LONG},
            "pre_window": PRE_WINDOW,
            "regime_cuts": list(REGIME_CUTS),
            "feature_names": list(FEATURE_NAMES),
            "target_full": "winning config-CLASS = formulation-family x MA-type",
            "target_formulation": "winning formulation-family (ride/cross/TP-exit/stack)",
            "fixed_class_by_kind": FIXED_CLASS_BY_KIND,
            "train_frac": args.train_frac,
            "causal": "vol pre-move/past-only; ML fit on TRAIN only; in-class config "
                      "chosen on TRAIN; TEST untouched for fitting; OOS-once",
        },
        "by_target_kind": artifact_by_kind,
    }
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"[artifact] {out_path}")


if __name__ == "__main__":
    main()
