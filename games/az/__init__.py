"""
chess_zero.az -- the AlphaZero (arXiv:1712.01815) frontier.

STRUCTURE + STUBS ONLY (no training run). The three pieces:
    net.py      -- residual CNN: board planes -> (policy logits, value scalar)
    mcts.py     -- PUCT-guided MCTS using the net's prior + value (no rollouts)
    selfplay.py -- self-play -> replay buffer -> train (loss = CE(policy)+MSE(value))

The classical ./engine.py is the strength reference these learned agents must
eventually beat. See ../README.md "Path to AlphaZero".
"""
