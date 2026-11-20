#!/bin/bash
# V22 / V25 post-fix training scripts (USE_CROSS_FEAT_ATTN=False default).
# Both versions tested + verified Capacity-tier on V22 (Ep12 ic1=+0.218, ShIC=0.000).
#
# Usage:
#   bash scripts/run_v22_v25_post_fix.sh v22         # V22 only
#   bash scripts/run_v22_v25_post_fix.sh v25         # V25 only
#   bash scripts/run_v22_v25_post_fix.sh both        # V22 then V25 sequential
#   bash scripts/run_v22_v25_post_fix.sh v22 5       # V22, 5 epochs only
#
# V22 ckpts: models/v22/base/v22_f29_wm_*.pt
#   - latest.pt = epoch_10.pt (training will resume from epoch 11)
#   - best_ema.pt is the post-fix Capacity-tier model (Ep12: ic1=+0.218,
#     ic16=+0.646, ic64=+0.605, ShIC=0.0000)
#
# V25 ckpts: models/v25/base/  (currently empty — fresh start expected)

set -e
cd "$(dirname "$0")/.."

VERSION=${1:-both}
MAX_EPOCHS=${2:-50}

run_v22() {
    echo "================================================================"
    echo "  V22 (iTransformer + patches + spectral + input_vib z=32"
    echo "       + USE_CROSS_FEAT_ATTN=False)"
    echo "  --max-epochs=$MAX_EPOCHS"
    echo "  Resuming from: $(ls -t models/v22/base/v22_f29_wm_latest.pt 2>/dev/null && echo present || echo NONE)"
    echo "================================================================"
    PYTHONIOENCODING=utf-8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python -u src/wm/v22/v22_training/train_world_model.py \
        --features 29 --resume --max-epochs "$MAX_EPOCHS" \
        2>&1 | tee "logs/v22/v22_post_fix_$(date +%Y%m%d_%H%M%S).log"
}

run_v25() {
    echo "================================================================"
    echo "  V25 (iTransformer + patches + spectral + period_emb + regime_ffn"
    echo "       + adv_regime + tail_Huber + USE_CROSS_FEAT_ATTN=False)"
    echo "  --max-epochs=$MAX_EPOCHS  --bf16 (default)"
    echo "================================================================"
    PYTHONIOENCODING=utf-8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python -u src/wm/v25/v25_training/train_world_model.py \
        --features 29 --max-epochs "$MAX_EPOCHS" --bf16 \
        2>&1 | tee "logs/v25/v25_post_fix_$(date +%Y%m%d_%H%M%S).log"
}

mkdir -p logs/v22 logs/v25

case "$VERSION" in
    v22)  run_v22 ;;
    v25)  run_v25 ;;
    both) run_v22 ; run_v25 ;;
    *)    echo "Usage: $0 [v22|v25|both] [max_epochs]" ; exit 1 ;;
esac

echo
echo "Done. Inspect ic1/ic16/ic64 at each epoch + ShIC every 10 epochs."
echo "Anti-fragile gate: ShIC/IC < 0.3 (V22 12-ep result was 0.000)."
echo "Capacity tier:     IC > 0.20 (V22 12-ep h=1 hit +0.218)."
