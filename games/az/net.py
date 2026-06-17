"""
chess_zero.az.net -- the AlphaZero residual CNN (arXiv:1712.01815).

Architecture (scaled down from the paper's 19/20 blocks x 256 filters so it
trains on a single consumer GPU; size is configurable via channels/n_blocks and
train_robust.py runs it at C=80/B=8 by default):

    input:  (B, 19, 8, 8) board planes  (see encoding.py)
    body:   conv-stem (3x3, C filters) + N residual blocks (2x conv + BN + skip)
    policy head: 1x1 conv -> flatten -> linear -> 4672 logits  (move plane scheme)
    value  head: 1x1 conv -> flatten -> linear -> hidden -> tanh scalar in [-1, 1]

Loss (computed in selfplay.py): cross-entropy(policy, MCTS-visit-distribution)
+ MSE(value, game-outcome z) + L2 weight decay -- exactly the paper's loss.

STATUS: LOAD-BEARING. This is the real network used by the full pipeline --
bootstrap_supervised.py (imitation pre-train) and train_robust.py (self-play ->
train -> eval loop with the champion gate + curriculum) both train these weights.
`__main__` still does a shape smoke test for a quick forward-pass sanity check.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import N_INPUT_PLANES, N_POLICY


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        return F.relu(x)


class PolicyHead(nn.Module):
    def __init__(self, channels: int, n_policy: int = N_POLICY):
        super().__init__()
        self.conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * 8 * 8, n_policy)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.flatten(1)
        return self.fc(x)  # raw logits; masked+softmaxed by the caller


class ValueHead(nn.Module):
    def __init__(self, channels: int, hidden: int = 256):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8 * 8, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))  # scalar in [-1, 1]


class AlphaZeroNet(nn.Module):
    """Dual-headed residual network: (planes) -> (policy_logits, value)."""

    def __init__(self, channels: int = 64, n_blocks: int = 6,
                 n_input_planes: int = N_INPUT_PLANES, n_policy: int = N_POLICY):
        super().__init__()
        self.stem_conv = nn.Conv2d(n_input_planes, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList(ResidualBlock(channels) for _ in range(n_blocks))
        self.policy_head = PolicyHead(channels, n_policy)
        self.value_head = ValueHead(channels)
        self.channels = channels
        self.n_blocks = n_blocks

    def forward(self, x):
        """x: (B, n_input_planes, 8, 8) -> (policy_logits (B, N_POLICY), value (B, 1))."""
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        for block in self.blocks:
            x = block(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

    @torch.no_grad()
    def predict(self, planes, legal_mask=None, device=None):
        """Convenience single-position inference.

        planes:     (N_INPUT_PLANES, 8, 8) numpy/torch.
        legal_mask: optional (N_POLICY,) {0,1}; illegal logits are masked to -inf.
        Returns (policy_probs np.ndarray (N_POLICY,), value float in [-1, 1]).
        """
        import numpy as np
        self.eval()
        device = device or next(self.parameters()).device
        x = torch.as_tensor(planes, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = self.forward(x)
        logits = logits.squeeze(0)
        if legal_mask is not None:
            m = torch.as_tensor(legal_mask, dtype=torch.bool, device=device)
            logits = logits.masked_fill(~m, float("-inf"))
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        probs = np.nan_to_num(probs)  # if a position had zero legal (shouldn't)
        return probs, float(value.item())

    @torch.no_grad()
    def predict_many(self, planes_list, masks_list=None, device=None):
        """BATCHED inference -- the throughput seam for parallel self-play. Evaluates a LIST of
        positions in ONE forward pass (the GPU is wildly underused at batch=1), so N games can
        batch their per-simulation leaf evals into a single launch.

        planes_list: list of (N_INPUT_PLANES, 8, 8) arrays (length B).
        masks_list:  optional list of (N_POLICY,) {0,1} legal masks (same length); illegal logits
                     -> -inf per position. Pass None to skip masking.
        Returns a list of (policy_probs (N_POLICY,), value float) -- SAME per-item contract as
        predict(), so a caller can swap predict() for predict_many() with no other change.
        Empty input -> []. NEVER trains (no_grad + eval)."""
        import numpy as np
        if not planes_list:
            return []
        self.eval()
        device = device or next(self.parameters()).device
        x = torch.as_tensor(np.asarray(planes_list), dtype=torch.float32, device=device)
        if x.dim() == 3:  # a single (planes,8,8) slipped through -> add batch dim
            x = x.unsqueeze(0)
        logits, value = self.forward(x)              # (B, N_POLICY), (B, 1)
        if masks_list is not None:
            m = torch.as_tensor(np.asarray(masks_list), dtype=torch.bool, device=device)
            logits = logits.masked_fill(~m, float("-inf"))
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        probs = np.nan_to_num(probs)
        vals = value.squeeze(-1).cpu().numpy()
        return [(probs[i], float(vals[i])) for i in range(probs.shape[0])]


class TicTacToeNet(nn.Module):
    """A tiny dual-head MLP for the 3x3 TicTacToe GameAdapter -- the minimal net that proves the
    GENERIC NeuralMCTS (mcts.NeuralMCTS) drives a NON-chess engine through the SAME contract as
    AlphaZeroNet. Input = (3, 3, 3) planes (my/their/side-to-move, from game_adapter.TicTacToe.encode);
    heads = 9 policy logits + a tanh value in [-1, 1]. Exposes the SAME predict()/predict_many()
    signature as AlphaZeroNet so NeuralMCTS consumes either with no change."""

    def __init__(self, n_input: int = 27, hidden: int = 64, n_policy: int = 9):
        super().__init__()
        self.n_policy = n_policy
        self.body = nn.Sequential(
            nn.Linear(n_input, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy_fc = nn.Linear(hidden, n_policy)
        self.value_fc = nn.Linear(hidden, 1)

    def forward(self, x):
        """x: (B, 3, 3, 3) or (B, 27) -> (policy_logits (B, 9), value (B, 1))."""
        x = x.flatten(1)
        h = self.body(x)
        return self.policy_fc(h), torch.tanh(self.value_fc(h))

    @torch.no_grad()
    def predict(self, planes, legal_mask=None, device=None):
        """Single-position inference; SAME contract as AlphaZeroNet.predict.
        planes: (3,3,3) or (27,) numpy/torch. legal_mask: optional (9,) {0,1}.
        Returns (policy_probs np.ndarray (9,), value float in [-1,1])."""
        import numpy as np
        self.eval()
        device = device or next(self.parameters()).device
        x = torch.as_tensor(planes, dtype=torch.float32, device=device).reshape(1, -1)
        logits, value = self.forward(x)
        logits = logits.squeeze(0)
        if legal_mask is not None:
            m = torch.as_tensor(legal_mask, dtype=torch.bool, device=device)
            logits = logits.masked_fill(~m, float("-inf"))
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        probs = np.nan_to_num(probs)
        return probs, float(value.item())

    @torch.no_grad()
    def predict_many(self, planes_list, masks_list=None, device=None):
        """Batched inference; SAME per-item contract as predict()."""
        import numpy as np
        if not planes_list:
            return []
        self.eval()
        device = device or next(self.parameters()).device
        x = torch.as_tensor(np.asarray([np.asarray(p).reshape(-1) for p in planes_list]),
                            dtype=torch.float32, device=device)
        logits, value = self.forward(x)
        if masks_list is not None:
            m = torch.as_tensor(np.asarray(masks_list), dtype=torch.bool, device=device)
            logits = logits.masked_fill(~m, float("-inf"))
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        probs = np.nan_to_num(probs)
        vals = value.squeeze(-1).cpu().numpy()
        return [(probs[i], float(vals[i])) for i in range(probs.shape[0])]


class Connect4Net(nn.Module):
    """A small conv tower for the 6x7 Connect-4 GameAdapter (game_adapter.Connect4 lives in
    az/connect4.py). Same dual-head design + SAME predict()/predict_many() API as AlphaZeroNet
    and TicTacToeNet, so the generic NeuralMCTS (mcts.NeuralMCTS) consumes it with NO change.

    Input  = (3, 6, 7) planes (my / their / side-to-move, from Connect4.encode).
    Body   = conv stem (3x3, C filters) + a few residual blocks over the 6x7 grid (spatial dims
             PRESERVED by padding=1 -- a 4-in-a-row is a local spatial pattern, so a conv tower is
             the right inductive bias, exactly as AlphaZero uses for Go/chess board games).
    Policy = 1x1 conv -> flatten -> linear -> 7 logits (one per column = the action space).
    Value  = 1x1 conv -> flatten -> linear -> tanh scalar in [-1, 1] (side-to-move POV).

    Small by design (CPU self-play under the 240s CI budget): C=32, 3 blocks ~ tens of k params.
    """

    def __init__(self, channels: int = 32, n_blocks: int = 3,
                 n_input_planes: int = 3, n_policy: int = 7, rows: int = 6, cols: int = 7):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.n_policy = n_policy
        self.stem_conv = nn.Conv2d(n_input_planes, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList(ResidualBlock(channels) for _ in range(n_blocks))
        # policy head: 1x1 conv to 2 planes -> flatten -> linear to n_policy logits
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * rows * cols, n_policy)
        # value head: 1x1 conv to 1 plane -> flatten -> linear -> hidden -> tanh
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(rows * cols, 64)
        self.value_fc2 = nn.Linear(64, 1)
        self.channels = channels
        self.n_blocks = n_blocks

    def forward(self, x):
        """x: (B, 3, 6, 7) -> (policy_logits (B, 7), value (B, 1) in [-1, 1])."""
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        for block in self.blocks:
            x = block(x)
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = self.policy_fc(p.flatten(1))               # raw logits; masked+softmaxed by the caller
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = F.relu(self.value_fc1(v.flatten(1)))
        v = torch.tanh(self.value_fc2(v))
        return p, v

    @torch.no_grad()
    def predict(self, planes, legal_mask=None, device=None):
        """Single-position inference; SAME contract as AlphaZeroNet.predict.
        planes: (3,6,7) numpy/torch. legal_mask: optional (7,) {0,1}.
        Returns (policy_probs np.ndarray (7,), value float in [-1,1])."""
        import numpy as np
        self.eval()
        device = device or next(self.parameters()).device
        x = torch.as_tensor(planes, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = self.forward(x)
        logits = logits.squeeze(0)
        if legal_mask is not None:
            m = torch.as_tensor(legal_mask, dtype=torch.bool, device=device)
            logits = logits.masked_fill(~m, float("-inf"))
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        probs = np.nan_to_num(probs)
        return probs, float(value.item())

    @torch.no_grad()
    def predict_many(self, planes_list, masks_list=None, device=None):
        """Batched inference; SAME per-item contract as predict()."""
        import numpy as np
        if not planes_list:
            return []
        self.eval()
        device = device or next(self.parameters()).device
        x = torch.as_tensor(np.asarray(planes_list), dtype=torch.float32, device=device)
        if x.dim() == 3:  # a single (3,6,7) slipped through -> add batch dim
            x = x.unsqueeze(0)
        logits, value = self.forward(x)
        if masks_list is not None:
            m = torch.as_tensor(np.asarray(masks_list), dtype=torch.bool, device=device)
            logits = logits.masked_fill(~m, float("-inf"))
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        probs = np.nan_to_num(probs)
        vals = value.squeeze(-1).cpu().numpy()
        return [(probs[i], float(vals[i])) for i in range(probs.shape[0])]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Shape smoke test (forward pass only; NO training).
    net = AlphaZeroNet(channels=64, n_blocks=6)
    print(f"AlphaZeroNet: channels=64 blocks=6  params={count_params(net):,}")
    dummy = torch.zeros(4, N_INPUT_PLANES, 8, 8)
    pol, val = net(dummy)
    print(f"policy logits: {tuple(pol.shape)}  (expect (4, {N_POLICY}))")
    print(f"value:         {tuple(val.shape)}  (expect (4, 1))  range~{val.min().item():.3f}..{val.max().item():.3f}")
