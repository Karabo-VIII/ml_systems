"""
chess_zero.az.visualize_selfplay -- SEE AlphaZero self-play FROM FIRST PRINCIPLES.

This is a transparency tool, not a black box. It plays ONE self-play game with the
trained AZ net + PUCT MCTS and records, for EVERY ply, the actual mechanics of the
AlphaZero learning loop:

    net proposes PRIORS  P(s,.)        (the policy head's raw guess, before search)
        |
        v
    MCTS SEARCH refines them into VISIT COUNTS  ->  pi  (the IMPROVED policy)
        |                                            (this is the policy TRAINING TARGET)
        v
    the VALUE head estimates the outcome  v in [-1,1]  (who's winning, side-to-move POV)
        |
        v
    the game's final result  z in {+1,0,-1}            (the value TRAINING TARGET)
        |
        v
    (board, pi, z) tuples are EXACTLY what trains the net  ->  the bootstrap.

The whole point of the side-by-side prior-vs-pi bar chart in the HTML is to let you
SEE search REFINE the prior: where pi differs from P, that delta is the information
MCTS discovered that the raw net did not yet know -- the engine teaching itself.

Outputs:
    selfplay_viz.html        -- self-contained, no-CDN, steppable (Prev/Next) viewer
    selfplay_viz_data.json   -- the raw per-ply first-principles data

Run (from the repo root):
    .venv/Scripts/python.exe projects/chess_zero/az/visualize_selfplay.py
    .venv/Scripts/python.exe projects/chess_zero/az/visualize_selfplay.py --terminal
    .venv/Scripts/python.exe projects/chess_zero/az/visualize_selfplay.py --sims 32 --max-plies 36

This is a SMALL bounded run by design (a few dozen MCTS sims/move, ~30-40 plies) so
it finishes in ~1-3 min on CPU/GPU. It demonstrates the MECHANICS; strength is
honestly compute-bounded (see ../README.md "RWYB results").
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import chess
import chess.svg

# Support both "python -m az.visualize_selfplay" and direct
# "python az/visualize_selfplay.py" invocation.
try:
    from .encoding import board_to_planes, legal_policy_mask, move_to_index
    from .net import AlphaZeroNet, count_params
    from .mcts import MCTS
except ImportError:  # direct-script invocation: make the package importable
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.abspath(os.path.join(_here, ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from az.encoding import (
        board_to_planes, legal_policy_mask, move_to_index)
    from az.net import AlphaZeroNet, count_params
    from az.mcts import MCTS


HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "az_demo_checkpoints")
HTML_PATH = os.path.join(HERE, "selfplay_viz.html")
JSON_PATH = os.path.join(HERE, "selfplay_viz_data.json")


# --------------------------------------------------------------------------- #
# Net loading: trained checkpoint if present, else a fresh (random) net.
# --------------------------------------------------------------------------- #
def load_net(device) -> Dict:
    """Return {net, source, channels, n_blocks, ckpt, params}.

    Prefers the highest-iter trained checkpoint in az_demo_checkpoints/; falls
    back to a fresh random-init net (priors ~ uniform) if none exist.
    """
    import torch

    ckpts = []
    if os.path.isdir(CKPT_DIR):
        for f in os.listdir(CKPT_DIR):
            if f.startswith("net_iter") and f.endswith(".pt"):
                try:
                    it = int(f[len("net_iter"):-len(".pt")])
                except ValueError:
                    continue
                ckpts.append((it, os.path.join(CKPT_DIR, f)))
    ckpts.sort()

    if ckpts:
        it, path = ckpts[-1]
        # train_demo.py saves {"state_dict","channels","n_blocks","iter"}.
        blob = torch.load(path, map_location=device, weights_only=False)
        if isinstance(blob, dict) and "state_dict" in blob:
            channels = int(blob.get("channels", 32))
            n_blocks = int(blob.get("n_blocks", 4))
            state = blob["state_dict"]
        else:
            # bare state_dict fallback
            channels, n_blocks, state = 32, 4, blob
        net = AlphaZeroNet(channels=channels, n_blocks=n_blocks).to(device)
        net.load_state_dict(state, strict=False)
        net.eval()
        return {
            "net": net, "source": "trained",
            "channels": channels, "n_blocks": n_blocks,
            "ckpt": os.path.basename(path), "iter": it,
            "params": count_params(net),
        }

    # Fresh net fallback.
    channels, n_blocks = 32, 4
    net = AlphaZeroNet(channels=channels, n_blocks=n_blocks).to(device)
    net.eval()
    return {
        "net": net, "source": "fresh",
        "channels": channels, "n_blocks": n_blocks,
        "ckpt": None, "iter": None,
        "params": count_params(net),
    }


# --------------------------------------------------------------------------- #
# Record one ply's first-principles data.
# --------------------------------------------------------------------------- #
def _top_moves_from_priors(board: chess.Board, priors: Dict[chess.Move, float],
                           k: int = 6) -> List[Dict]:
    items = sorted(priors.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [{"uci": mv.uci(), "san": board.san(mv), "p": float(p)}
            for mv, p in items]


def _top_moves_from_visits(board: chess.Board, visits: Dict[chess.Move, int],
                           priors: Dict[chess.Move, float], k: int = 6) -> List[Dict]:
    total = sum(visits.values()) or 1
    items = sorted(visits.items(), key=lambda kv: kv[1], reverse=True)[:k]
    out = []
    for mv, n in items:
        out.append({
            "uci": mv.uci(),
            "san": board.san(mv),
            "visits": int(n),
            "pi": float(n) / total,          # the improved-policy probability
            "prior": float(priors.get(mv, 0.0)),  # raw net prior for the SAME move
        })
    return out


def evaluate_priors_value(net, board: chess.Board, device):
    """Raw net forward pass -> (priors_by_move, value) WITHOUT any search.
    Mirrors MCTS._evaluate so the displayed prior is exactly what seeds search."""
    planes = board_to_planes(board)
    mask, idx_to_move = legal_policy_mask(board)
    probs, value = net.predict(planes, legal_mask=mask, device=device)
    priors = {mv: float(probs[idx]) for idx, mv in idx_to_move.items()}
    total = sum(priors.values())
    if total > 0:
        priors = {m: p / total for m, p in priors.items()}
    else:
        n = max(1, len(idx_to_move))
        priors = {m: 1.0 / n for m in idx_to_move}
    return priors, float(value)


# --------------------------------------------------------------------------- #
# Play one self-play game, recording every ply.
# --------------------------------------------------------------------------- #
def play_and_record(net, device, n_sims: int, max_plies: int,
                    temp_moves: int, c_puct: float, seed: int,
                    terminal: bool) -> Dict:
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)

    mcts = MCTS(net, c_puct=c_puct, n_simulations=n_sims, device=device)
    board = chess.Board()
    plies: List[Dict] = []
    ply = 0

    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        temperature = 1.0 if ply < temp_moves else 0.0
        fen = board.fen()
        side = "white" if board.turn == chess.WHITE else "black"

        # 1) RAW net read (priors + value) -- before search.
        priors, value = evaluate_priors_value(net, board, device)

        # 2) MCTS search -> visit counts (the improved policy pi).
        visits = mcts.run(board, add_noise=(temperature > 0))

        # 3) Pick the move the way self-play does (temp sampling early, greedy late).
        moves = list(visits.keys())
        counts = np.array([visits[m] for m in moves], dtype=np.float64)
        if counts.sum() == 0:
            played = moves[0]
        elif temperature <= 1e-6:
            played = moves[int(counts.argmax())]
        else:
            p = counts ** (1.0 / temperature)
            p = p / p.sum()
            played = moves[int(np.random.choice(len(moves), p=p))]

        prior_top = _top_moves_from_priors(board, priors, k=6)
        pi_top = _top_moves_from_visits(board, visits, priors, k=6)
        svg = chess.svg.board(board, size=360,
                              lastmove=(board.peek() if board.move_stack else None))

        rec = {
            "ply": ply,
            "move_number": board.fullmove_number,
            "side_to_move": side,
            "fen": fen,
            "svg": svg,
            "net_prior_top": prior_top,        # raw policy prior, top-6 legal
            "mcts_pi_top": pi_top,             # MCTS visit dist, top-6 (+ prior for same move)
            "net_value": value,                # v in [-1,1], side-to-move POV
            "n_sims": n_sims,
            "temperature": temperature,
            "played_uci": played.uci(),
            "played_san": board.san(played),
            "player_to_score": side,           # used to assign z after the game
            "z": None,                         # filled in after the game ends
        }
        plies.append(rec)

        if terminal:
            _print_ply_terminal(board, rec)

        board.push(played)
        ply += 1

    # Outcome + value target z for each ply (from that ply's mover's POV).
    result = board.result(claim_draw=True)
    if result == "1-0":
        winner = "white"
    elif result == "0-1":
        winner = "black"
    else:
        winner = None
    for rec in plies:
        if winner is None:
            rec["z"] = 0.0
        else:
            rec["z"] = 1.0 if rec["player_to_score"] == winner else -1.0

    termination = _termination_reason(board)
    return {
        "plies": plies,
        "result": result,
        "winner": winner,
        "termination": termination,
        "final_fen": board.fen(),
        "n_plies": len(plies),
        "hit_ply_cap": ply >= max_plies and not board.is_game_over(claim_draw=True),
    }


def _termination_reason(board: chess.Board) -> str:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient_material"
    if board.can_claim_fifty_moves() or board.is_fifty_moves():
        return "fifty_move_rule"
    if board.can_claim_threefold_repetition() or board.is_repetition(3):
        return "threefold_repetition"
    if board.is_seventyfive_moves():
        return "seventyfive_move_rule"
    if board.is_fivefold_repetition():
        return "fivefold_repetition"
    return "ply_cap_reached"


# --------------------------------------------------------------------------- #
# --terminal: ASCII board + top-3 MCTS moves + value, per ply.
# --------------------------------------------------------------------------- #
def _ascii_board(board: chess.Board) -> str:
    """cp1252-safe ASCII board (UPPER=white, lower=black, '.'=empty), rank 8 at top.
    Avoids board.unicode() whose chess glyphs crash the Windows cp1252 console."""
    lines = []
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            cells.append(piece.symbol() if piece else ".")
        lines.append(f"{rank + 1}  " + " ".join(cells))
    lines.append("   a b c d e f g h")
    return "\n".join(lines)


def _print_ply_terminal(board: chess.Board, rec: Dict) -> None:
    print("\n" + "=" * 60)
    print(f"PLY {rec['ply']}  (move {rec['move_number']}, {rec['side_to_move']} to move)"
          f"  temp={rec['temperature']:.0f}")
    print("-" * 60)
    print(_ascii_board(board))
    print("-" * 60)
    v = rec["net_value"]
    who = rec["side_to_move"]
    lean = "winning" if v > 0.15 else ("losing" if v < -0.15 else "~even")
    print(f"net VALUE v = {v:+.3f}  ({who} to move sees itself {lean})")
    print("top-3 MCTS moves (visit% = improved policy pi; prior = raw net):")
    for m in rec["mcts_pi_top"][:3]:
        print(f"   {m['san']:8s}  pi={m['pi']*100:5.1f}%  "
              f"(prior {m['prior']*100:5.1f}%, {m['visits']} visits)")
    print(f"PLAYED: {rec['played_san']}")


# --------------------------------------------------------------------------- #
# HTML renderer (self-contained, no external deps / CDN; vanilla JS stepper).
# --------------------------------------------------------------------------- #
def render_html(game: Dict, meta: Dict) -> str:
    data_js = json.dumps(game["plies"], separators=(",", ":"))
    meta_js = json.dumps(meta, separators=(",", ":"))
    game_js = json.dumps({k: game[k] for k in
                          ("result", "winner", "termination", "n_plies",
                           "hit_ply_cap", "final_fen")},
                         separators=(",", ":"))

    net_line = (f"TRAINED net (checkpoint {meta['ckpt']}, iter {meta['iter']})"
                if meta["source"] == "trained"
                else "FRESH random-init net (no checkpoint found; priors ~ uniform)")

    # Note: no f-string for the big template -- it contains many { } for CSS/JS.
    # We inject via .replace() on unambiguous tokens instead.
    html = HTML_TEMPLATE
    html = html.replace("__DATA_JS__", data_js)
    html = html.replace("__META_JS__", meta_js)
    html = html.replace("__GAME_JS__", game_js)
    html = html.replace("__NET_LINE__", _esc(net_line))
    html = html.replace("__DEVICE__", _esc(meta["device"]))
    html = html.replace("__NPLIES__", str(game["n_plies"]))
    html = html.replace("__RESULT__", _esc(game["result"]))
    html = html.replace("__TERMINATION__", _esc(game["termination"]))
    html = html.replace("__SIMS__", str(meta["n_sims"]))
    return html


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AlphaZero self-play -- from first principles</title>
<style>
  :root {
    --bg:#0f1117; --panel:#1a1d29; --panel2:#232735; --ink:#e8eaf0;
    --muted:#9aa3b8; --prior:#5b8def; --pi:#36c08a; --accent:#f0a93b;
    --good:#36c08a; --bad:#e0556b; --line:#333a4d;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         line-height:1.45; }
  header { padding:18px 22px; background:var(--panel); border-bottom:1px solid var(--line); }
  header h1 { margin:0 0 6px; font-size:20px; }
  header p { margin:4px 0; color:var(--muted); font-size:13px; max-width:980px; }
  .loop { font-size:12.5px; color:var(--ink); background:var(--panel2);
          border:1px solid var(--line); border-radius:8px; padding:10px 14px;
          margin-top:10px; max-width:980px; }
  .loop b { color:var(--accent); }
  .loop code { color:var(--pi); }
  .wrap { display:flex; gap:22px; padding:22px; flex-wrap:wrap; align-items:flex-start; }
  .col-board { flex:0 0 auto; }
  .col-data { flex:1 1 420px; min-width:380px; }
  .board { background:var(--panel); border:1px solid var(--line); border-radius:10px;
           padding:14px; display:inline-block; }
  .board svg { display:block; }
  .controls { margin-top:14px; display:flex; gap:10px; align-items:center; }
  button { background:var(--panel2); color:var(--ink); border:1px solid var(--line);
           border-radius:8px; padding:9px 16px; font-size:14px; cursor:pointer; }
  button:hover:not(:disabled) { background:#2d3346; }
  button:disabled { opacity:.4; cursor:not-allowed; }
  .ply-label { font-size:14px; color:var(--muted); margin-left:auto; }
  .meta-box { background:var(--panel); border:1px solid var(--line); border-radius:10px;
              padding:12px 16px; margin-bottom:14px; font-size:13px; }
  .meta-box .row { display:flex; justify-content:space-between; gap:12px; padding:2px 0; }
  .meta-box .row span:first-child { color:var(--muted); }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px;
          padding:14px 16px; margin-bottom:14px; }
  .card h3 { margin:0 0 4px; font-size:14px; }
  .card .sub { color:var(--muted); font-size:12px; margin:0 0 12px; }
  .legend { font-size:12px; color:var(--muted); margin-bottom:10px; }
  .swatch { display:inline-block; width:11px; height:11px; border-radius:2px;
            vertical-align:middle; margin:0 4px 0 12px; }
  .sw-prior { background:var(--prior); }
  .sw-pi { background:var(--pi); }
  .barrow { margin:9px 0; }
  .barrow .mv { font-size:13px; margin-bottom:3px; display:flex; justify-content:space-between; }
  .barrow .mv .san { font-weight:600; }
  .barrow .mv .delta { font-size:11px; }
  .delta.up { color:var(--good); } .delta.down { color:var(--bad); }
  .track { position:relative; height:30px; background:var(--panel2);
           border-radius:6px; overflow:hidden; }
  .bar { position:absolute; left:0; height:14px; border-radius:4px;
         display:flex; align-items:center; padding-left:6px; font-size:10.5px;
         color:#0b0d12; font-weight:700; white-space:nowrap; transition:width .12s; }
  .bar.prior { top:1px; background:var(--prior); }
  .bar.pi { bottom:1px; background:var(--pi); }
  .value-gauge { margin-top:6px; }
  .gauge-track { position:relative; height:26px; background:linear-gradient(
      90deg, var(--bad) 0%, #4a4f63 50%, var(--good) 100%);
      border-radius:6px; }
  .gauge-mid { position:absolute; left:50%; top:-3px; bottom:-3px; width:2px;
               background:var(--muted); }
  .gauge-knob { position:absolute; top:-4px; width:0; height:0;
                border-left:7px solid transparent; border-right:7px solid transparent;
                border-top:10px solid #fff; transform:translateX(-7px); transition:left .12s; }
  .gauge-labels { display:flex; justify-content:space-between; font-size:11px;
                  color:var(--muted); margin-top:4px; }
  .played { font-size:15px; }
  .played b { color:var(--accent); font-size:18px; }
  .ztag { font-size:12px; color:var(--muted); margin-top:6px; }
  .ztag b.win { color:var(--good); } .ztag b.loss { color:var(--bad); }
  .ztag b.draw { color:var(--muted); }
  code { background:var(--panel2); padding:1px 5px; border-radius:4px; font-size:12px; }
  .foot { padding:8px 22px 26px; color:var(--muted); font-size:12px; }
</style>
</head>
<body>
<header>
  <h1>AlphaZero self-play &mdash; from first principles</h1>
  <p>One real self-play game by the project's AZ net (__NET_LINE__) on <b>__DEVICE__</b>,
     with PUCT&nbsp;MCTS at <b>__SIMS__</b> simulations/move. Result: <b>__RESULT__</b>
     (__TERMINATION__), <b>__NPLIES__</b> plies. Step with Prev/Next.</p>
  <div class="loop">
    The first-principles loop &mdash; <b>net proposes priors</b> P(s,&middot;) &rarr;
    <b>MCTS search refines</b> them into visit-counts =
    the policy target <code>&pi;</code> &rarr; the <b>value head</b> estimates the
    outcome <code>v</code> &rarr; the game's <b>final result</b> <code>z</code> is the
    value target &rarr; the <code>(board,&nbsp;&pi;,&nbsp;z)</code> tuples are
    <b>EXACTLY what trains the net</b>. Watch the green <code>&pi;</code> bars diverge
    from the blue prior bars: that delta is what <i>search discovered</i> &mdash; the
    engine teaching itself.
  </div>
</header>

<div class="wrap">
  <div class="col-board">
    <div class="board" id="board"></div>
    <div class="controls">
      <button id="prev">&larr; Prev</button>
      <button id="next">Next &rarr;</button>
      <span class="ply-label" id="plyLabel"></span>
    </div>
  </div>

  <div class="col-data">
    <div class="meta-box" id="metaBox"></div>

    <div class="card">
      <h3>Policy: raw prior vs. MCTS-refined &pi;</h3>
      <p class="sub">Top moves by MCTS visit count. <span class="swatch sw-prior"></span>
        blue = net's RAW prior P(s,a); <span class="swatch sw-pi"></span>
        green = &pi; = MCTS visit% (the policy training target). The arrow shows how
        search moved the probability.</p>
      <div id="policyBars"></div>
    </div>

    <div class="card">
      <h3>Value head: v (who's winning, side-to-move POV)</h3>
      <p class="sub">v &isin; [&minus;1, +1]. &minus;1 = side to move is lost,
        +1 = side to move is winning.</p>
      <div class="value-gauge">
        <div class="gauge-track">
          <div class="gauge-mid"></div>
          <div class="gauge-knob" id="gaugeKnob"></div>
        </div>
        <div class="gauge-labels"><span>&minus;1 losing</span><span id="vNum">0</span><span>+1 winning</span></div>
      </div>
    </div>

    <div class="card">
      <div class="played" id="played"></div>
      <div class="ztag" id="ztag"></div>
    </div>
  </div>
</div>

<div class="foot">
  Self-contained: board is inline SVG from <code>chess.svg.board()</code>; bars are
  plain CSS divs; all per-ply data embedded as a JS array (no network, no CDN).
  Strength is honestly compute-bounded &mdash; this visualizes the MECHANICS, not a
  super-human engine.
</div>

<script>
const PLIES = __DATA_JS__;
const META  = __META_JS__;
const GAME  = __GAME_JS__;
let i = 0;

function pct(x){ return (x*100).toFixed(1) + "%"; }

function render(){
  const p = PLIES[i];

  // board
  document.getElementById("board").innerHTML = p.svg;

  // ply label
  document.getElementById("plyLabel").textContent =
    "ply " + p.ply + " / " + (PLIES.length - 1) + "  (move " + p.move_number + ")";

  // meta box
  const tempTxt = p.temperature >= 0.5
    ? "1 (exploration: sample &prop; visits)"
    : "0 (greedy: argmax visits)";
  document.getElementById("metaBox").innerHTML =
    '<div class="row"><span>side to move</span><b>' + p.side_to_move + '</b></div>' +
    '<div class="row"><span>MCTS sims this move</span><b>' + p.n_sims + '</b></div>' +
    '<div class="row"><span>self-play temperature</span><b>' + tempTxt + '</b></div>' +
    '<div class="row"><span>FEN</span><code style="font-size:11px">' + p.fen + '</code></div>';

  // policy bars: pi (green) vs prior (blue) for the same top moves
  const rows = p.mcts_pi_top;
  let maxv = 0.0001;
  rows.forEach(r => { maxv = Math.max(maxv, r.pi, r.prior); });
  let html = "";
  rows.forEach(r => {
    const dv = r.pi - r.prior;
    const arrow = dv > 0.005 ? '<span class="delta up">&uarr; +' + pct(dv) + ' from search</span>'
                : dv < -0.005 ? '<span class="delta down">&darr; ' + pct(dv) + ' from search</span>'
                : '<span class="delta">&middot;</span>';
    const wpi    = Math.max(2, (r.pi / maxv) * 100);
    const wprior = Math.max(2, (r.prior / maxv) * 100);
    html +=
      '<div class="barrow">' +
        '<div class="mv"><span class="san">' + r.san + '</span>' + arrow + '</div>' +
        '<div class="track">' +
          '<div class="bar prior" style="width:' + wprior + '%">P ' + pct(r.prior) + '</div>' +
          '<div class="bar pi" style="width:' + wpi + '%">&pi; ' + pct(r.pi) + '</div>' +
        '</div>' +
      '</div>';
  });
  document.getElementById("policyBars").innerHTML = html;

  // value gauge: map v in [-1,1] -> [0,100]%
  const v = p.net_value;
  const left = ((v + 1) / 2) * 100;
  document.getElementById("gaugeKnob").style.left = left + "%";
  document.getElementById("vNum").textContent = "v = " + (v>=0?"+":"") + v.toFixed(3);

  // played move
  document.getElementById("played").innerHTML =
    "move played: <b>" + p.played_san + "</b> <code>" + p.played_uci + "</code>";

  // z target
  let zhtml = "";
  if (p.z === null || p.z === undefined) {
    zhtml = "value target z: (pending game end)";
  } else {
    let cls = p.z > 0 ? "win" : (p.z < 0 ? "loss" : "draw");
    let word = p.z > 0 ? "this side WON" : (p.z < 0 ? "this side LOST" : "DREW");
    zhtml = 'value training target for this ply: <b class="' + cls + '">z = ' +
            (p.z>=0?"+":"") + p.z.toFixed(0) + '</b> (' + word +
            ', game ' + GAME.result + ' by ' + GAME.termination + ')';
  }
  document.getElementById("ztag").innerHTML = zhtml;

  document.getElementById("prev").disabled = (i === 0);
  document.getElementById("next").disabled = (i === PLIES.length - 1);
}

document.getElementById("prev").onclick = () => { if (i>0){ i--; render(); } };
document.getElementById("next").onclick = () => { if (i<PLIES.length-1){ i++; render(); } };
document.addEventListener("keydown", e => {
  if (e.key === "ArrowLeft")  { if (i>0){ i--; render(); } }
  if (e.key === "ArrowRight") { if (i<PLIES.length-1){ i++; render(); } }
});
render();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Visualize one AlphaZero self-play game from first principles.")
    ap.add_argument("--sims", type=int, default=32,
                    help="MCTS simulations per move (default 32; small+bounded).")
    ap.add_argument("--max-plies", type=int, default=36,
                    help="hard cap on plies so the run stays ~1-3 min (default 36).")
    ap.add_argument("--temp-moves", type=int, default=12,
                    help="first N plies sampled with temperature=1 (exploration).")
    ap.add_argument("--c-puct", type=float, default=1.5, help="PUCT exploration constant.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed.")
    ap.add_argument("--terminal", action="store_true",
                    help="also print each ply to the console (ASCII board + top-3 + value).")
    args = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = (torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")

    info = load_net(device)
    print(f"[device] {device} ({dev_name})")
    if info["source"] == "trained":
        print(f"[net] TRAINED checkpoint {info['ckpt']} (iter {info['iter']}) | "
              f"channels={info['channels']} blocks={info['n_blocks']} "
              f"params={info['params']:,}")
    else:
        print(f"[net] FRESH random-init net (no checkpoint found in az_demo_checkpoints/) | "
              f"channels={info['channels']} blocks={info['n_blocks']} "
              f"params={info['params']:,} -- priors will be ~uniform")
    print(f"[run] sims/move={args.sims}  max_plies={args.max_plies}  "
          f"temp_moves={args.temp_moves}  c_puct={args.c_puct}  seed={args.seed}")

    t0 = time.time()
    game = play_and_record(
        info["net"], device, n_sims=args.sims, max_plies=args.max_plies,
        temp_moves=args.temp_moves, c_puct=args.c_puct, seed=args.seed,
        terminal=args.terminal)
    dt = time.time() - t0

    meta = {
        "source": info["source"],
        "ckpt": info["ckpt"],
        "iter": info["iter"],
        "channels": info["channels"],
        "n_blocks": info["n_blocks"],
        "params": info["params"],
        "device": f"{device} ({dev_name})",
        "n_sims": args.sims,
        "max_plies": args.max_plies,
        "temp_moves": args.temp_moves,
        "c_puct": args.c_puct,
        "seed": args.seed,
        "wall_s": round(dt, 1),
        "n_input_planes": 19,
        "n_policy": 4672,
    }

    # Write JSON (raw per-ply first-principles data).
    out = {"meta": meta,
           "game": {k: game[k] for k in
                    ("result", "winner", "termination", "final_fen",
                     "n_plies", "hit_ply_cap")},
           "plies": game["plies"]}
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Write HTML.
    html = render_html(game, meta)
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print("\n" + "=" * 60)
    print(f"[done] {game['n_plies']} plies recorded in {dt:.1f}s "
          f"({game['result']} by {game['termination']})")
    if game["hit_ply_cap"]:
        print(f"[note] game hit the ply cap ({args.max_plies}) before a natural end "
              f"-- z assigned from the (unfinished) result '{game['result']}'.")
    print(f"[out] HTML -> {os.path.relpath(HTML_PATH, HERE)}  "
          f"(open projects/chess_zero/az/selfplay_viz.html in a browser)")
    print(f"[out] JSON -> {os.path.relpath(JSON_PATH, HERE)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
