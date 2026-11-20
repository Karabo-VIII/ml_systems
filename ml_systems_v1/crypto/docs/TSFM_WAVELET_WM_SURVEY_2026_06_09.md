# Time-series foundation models + wavelets for crypto, and the WM layer (survey + recommendation, 2026-06-09)

Triggered by the user (Stream C of the 6h mandate): *"there are open source engines for understanding the
market ... time series open source libraries/models ... for crypto. similar to the chess engine ... understand
the market at a fundamental level, including say, wavelet theory ... this touches the WM layer. so look into
that extensively."*

**Verdict up front (read this, skip the rest if busy):** We are NOT behind on time-series foundation models —
we already evaluated the best finance-specific one (Kronos) and it **lost to our own V1.1 model** (zero-shot
pooled IC 0.029 vs our 0.067). The genuine, untouched gap is **wavelets/multi-scale spectral features (zero
usage in the repo)** — BUT the entire published wavelet-trading literature is contaminated by look-ahead, so the
only honest way to adopt it is **causal-only, as a FEATURE for the existing setup/oracle apparatus, evaluated on
capture-rate/compound (NOT IC, NOT standalone forecasting)**. The highest-EV move is a small, leak-guarded causal
wavelet feature probe run through the firewall + synthetic-positive-control we already built — not a new model.

---

## 1. The OSS landscape (what exists, 2025-2026)

Time-series foundation models (TSFMs) — pretrained transformers for zero-shot/few-shot forecasting — matured a lot
in 2025:

| Model | Origin | Shape | Finance fit |
|---|---|---|---|
| **Kronos** | Tsinghua/NeoQuasar, AAAI 2026 | decoder-only AR on **OHLCV K-line tokens**, 12B K-lines / 45 exchanges, MIT, HF | **The only one purpose-built on candlesticks.** mini/small/base/large (4M-499M). |
| **TimesFM** | Google | decoder-only + patching, 100B time-points / 200M params | general; battle-tested in Google prod |
| **Chronos-2** | Amazon | tokenize→encoder, natively probabilistic; Oct-2025 added multivariate+covariates | largest community, easiest to prototype |
| **Moirai-2** | Salesforce | any-variate attention (cross-series from the start) | best when cross-asset structure matters |
| **MOMENT** | CMU AutonLab | encoder, masked-reconstruction; embeddings + anomaly | **descriptive** (analogs/anomaly), not forecasting |
| **Lag-Llama / TinyTimeMixers / TOTO** | various | small probabilistic / lightweight | edge/cheap baselines |

Two caveats the literature itself flags, both load-bearing for us:
- **Benchmark contamination**: TSFM evals routinely overlap pretraining and test data, inflating reported accuracy
  by **47-184%**. Any zero-shot number from a paper is suspect until reproduced on *our* unseen segment.
- **"BERT moment? — not yet for finance"** (NeurIPS 2025 workshop): zero-shot TSFMs do NOT reliably beat a
  domain model fine-tuned on the target; the consistent finding is *fewer years of data to reach parity*, not
  *higher ceiling*.

Sources: [ML-Mastery 2026 toolkit](https://machinelearningmastery.com/the-2026-time-series-toolkit-5-foundation-models-for-autonomous-forecasting/),
[PapersWithBacktest TimesFM/Chronos/MOIRAI](https://paperswithbacktest.com/course/timesfm-vs-chronos-vs-moirai),
[Kronos (arXiv 2508.02739)](https://arxiv.org/html/2508.02739v1),
[Kronos repo](https://github.com/shiyu-coder/Kronos),
[TSFM in finance survey (ACM 2025)](https://dl.acm.org/doi/full/10.1145/3785706.3785728).

## 2. What WE already have (the part that stops us re-inventing)

The WM layer already integrates this class of model — this is not a greenfield:

- **V24 (TimesNet)** `src/wm/v24/timesnet_backbone.py:70-86` — native `torch.fft.rfft` top-K period detection +
  2D-inception. We already do spectral period decomposition inside a WM (no external dep).
- **Kronos baseline** `src/frontier_ml/kronos_baseline/` — we ran Kronos-small **zero-shot** on chimera_legacy
  u10 OOS. **Result (`logs/frontier_ml/kronos_baseline/kronos_small_zero_shot_*.json`): pooled IC h=1 = 0.029**
  (per-asset −0.10..+0.18, n=100 each = noisy), vs **our V1.1 record 0.067** and the 0.10 headline. The
  pre-registered decision rule (≥0.060 → pivot to Kronos finetune; <0.030 → Kronos doesn't help) landed at
  **"doesn't help — stack stays on plan."** A 12B-K-line foundation model scored *below our 8GB-trained model*
  on our data, zero-shot.
- **MOMENT-1** `src/narrate/foundation.py` — used in the DESCRIPTIVE layer (anomaly percentiles / historical
  analogs), not forecasting. Correct use.
- **V25** `src/wm/v25/` — first-principles crypto synthesis (hard-coded period embeddings + regime-conditioned
  attention) — our own answer to "TSFM but crypto-aware."

**Implication:** the "are we missing a market-understanding engine?" question is largely answered *no* for TSFMs.
We tested the strongest candidate and it lost. What we have NOT done is the part below.

## 3. The genuine gap: wavelets / multi-scale (zero usage) — and its trap

Grep confirms **zero** `pywt` / `wavelet` / `scipy.signal` usage anywhere in `src/wm/` or `src/pipeline/` (V24's
FFT is the only spectral code). So wavelets are a real, untouched axis. They are also a natural fit for the
project's founding framing — **the unit of trading is a SETUP across a MULTI-CANDLE MOVE** (MEMORY.md). A wavelet
decomposition answers exactly *"at which time-scale is energy/structure concentrated right now, and is it
expanding?"* — i.e. it is a multi-scale description of a *move*, not a per-candle prediction.

**But the literature is a trap.** The Wavelet-LSTM papers reporting 31% RMSE reductions are, in the large majority,
**look-ahead-contaminated**:
- They apply a discrete wavelet transform (DWT) to the **entire series before the train/test split**, so the
  denoised "past" already contains future bars (the boundary coefficients depend on future samples).
- They use **non-causal symmetric** wavelets whose reconstruction at time t uses t+k.

This is the **same bug class as our G-AUDIT-011 (look-ahead via full-history standardization)** and exactly what
our `leak_guard` / `firewall` machinery exists to catch. Honest take: **discard the published OOS gains as
mostly leakage.** Wavelets are valid for us ONLY as a **causal, past-only** transform (rolling/expanding-window
MODWT or à-trous, computed with no future bar), and any gain MUST clear our own leak-guarded gate before belief.

Sources: [Wavelet-LSTM denoising review](https://cmpublisher.com/wavelet-transforms-in-financial-time-series-analysis-a-review-on-stock-price-prediction/),
[overfitting/parameter-tuning caveats](https://medium.com/@amit25173/using-wavelet-transforms-in-time-series-forecasting-aeca30204ea2),
[denoising + LSTM (arXiv 2103.03505)](https://arxiv.org/pdf/2103.03505). Lib: PyWavelets (`pywt`, `swt`/`modwt` for
the shift-invariant causal variants), `ssqueezepy` (synchrosqueezing).

## 4. The integration surface (from the WM-layer recon)

Two clean insertion points (neither requires touching WM architecture for option A):

- **(A) Causal spectral/wavelet FEATURE block → pipeline.** New `src/pipeline/features/spectral_features.py`
  emitting `spec_*` columns (per-bar, rolling-window, past-only); register in `config/feature_registry.yaml`;
  join in `src/pipeline/make_dataset.py`; new `f<N>` family in `src/feature_sets.py`. The WM consumes it
  transparently (`forward_train([B,T,F])`, F grows). **This is the low-risk path.**
- **(B) TSFM as feature-extractor or encoder.** (B1) pre-compute Chronos/TimesFM embeddings per 96-bar window →
  `tsfm_*` columns (PCA to 32-64 dims) → same feature pipeline. (B2) deeper: a `src/wm/v26/` with a frozen TSFM
  encoder + our return/regime heads. **Higher cost, and Kronos already suggests low payoff for zero-shot
  forecasting** — only worth it as embeddings feeding our objective, not as a standalone forecaster.

## 5. Recommendation (tied to the current framing, EV-ranked)

The project's objective is **robust held-out COMPOUND return on setups across moves** — IC/per-bar predictability
is banned as a primary metric (MEMORY.md). That re-frames everything above:

1. **[HIGHEST EV, LOW RISK] Causal wavelet *feature*, evaluated on capture-rate, gated by our soundness spine.**
   Build option-A with a past-only rolling MODWT producing 3-4 features: per-scale energy, the dominant scale,
   and scale-energy *expansion* (a multi-scale generalization of the vol-expansion trigger C1 just probed). Then
   — and this is the non-negotiable part — run it through `src/strat/firewall.py` (random-entry + regime/
   membership-matched null), the `synthetic_positive_control` (prove the gate can ACCEPT a real signal, two-sided
   soundness), and measure whether it lifts the **oracle capture-rate** (`src/oracle/`), NOT IC. Asymmetric loss:
   default to "no edge" unless it clears the leak-guarded gate.
2. **[MEDIUM EV] Re-evaluate one TSFM (Chronos-2) the RIGHT way** — not zero-shot IC (Kronos already failed that),
   but its *embeddings* as features into the setup/capture apparatus, on the compound objective. Only if (1)
   shows multi-scale features help (i.e. there is signal a pretrained encoder might capture better).
3. **[LOW EV / DO NOT] A new V26 TSFM-encoder WM, or chasing zero-shot TSFM forecasting.** Kronos (12B K-lines,
   finance-specific) scored 0.029 < our 0.067. Re-paying for that is the over-mining trap. Park unless (1)+(2)
   surface a concrete reason.

**The honest one-liner:** wavelets are the real new axis, but they are a *feature for our existing
setup/oracle/soundness machinery measured on capture-rate*, not a new forecasting engine — and they must be
causal + leak-guarded or they will produce the same fake OOS gains that fill the literature.

## 6. What was verified vs asserted (provenance)

- Kronos 0.029 / V1.1 0.067 / headline 0.10: **VERIFIED** — read from
  `logs/frontier_ml/kronos_baseline/kronos_small_zero_shot_1777724420.json` (rows + pooled) + `eval_kronos.py`
  decision rule.
- Zero wavelet usage: **VERIFIED** — grep `pywt|wavelet|scipy.signal` across `src/wm` + `src/pipeline` (only
  V24 `torch.fft`).
- WM input/output contract + integration points: **VERIFIED** — WM-layer structural recon (feature_sets.py,
  make_dataset.py, chimera_loader.py, world_model.py forward_train).
- TSFM landscape + contamination 47-184% + wavelet look-ahead: **REPORTED** (web, 2026-06-09) — cited above;
  the look-ahead critique is corroborated by our own G-AUDIT-011 invariant.
