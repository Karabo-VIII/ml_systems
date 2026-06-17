"""Prong 2 — distillation from V1.x ensemble + foundation into a 5-10M
deployable student.

Per PLAN.md: ensemble inference at 8x cost is too slow for live trading.
Distill diversity into a single student that runs at 1/8th the cost
while matching ensemble IC (target: IC >= 0.95 * best_teacher_IC AND
latency <= 1/4 ensemble inference time).

Workflow:
    1. teacher_inference.py -- per-asset, per-window, run every teacher
       model (V1.x, V3, V4, V6, V11/12/14, foundation) and CACHE their
       TwoHot logits + h_seq mean-pool to disk. One-time cost.
    2. student.py -- 5-10M FoundationBackbone-style student model.
    3. distill_loss.py -- hybrid alpha*KL + beta*L1(expectation) +
       gamma*L2(variance) per LITERATURE.md Hole 5.
    4. train.py -- training loop: load cached teacher logits + student
       forward + hybrid loss + AMP + ckpt-every-100 + harmony.

All teachers are FROZEN. Only the student trains.
"""
