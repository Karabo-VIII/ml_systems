"""
chess_zero.az.live_viz -- a TRUE browser visualizer for the learn-watch loop.

The terminal ASCII board is NOT a real visualizer. This module writes a
self-contained HTML page (out_dir/live.html) showing the chess board with REAL
piece graphics (python-chess `chess.svg.board()` -> inline SVG, no CDN), the SAN
move list, a header (matchup + which iter the live net is from), and the strength
curve (winrate_vs_random / winrate_vs_classical over iters) as an inline SVG
sparkline + a small table.

How it stays a LIVE view with NO server:
    * the page carries  <meta http-equiv="refresh" content="1">  so the browser
      re-reads the file once a second;
    * every .update() rewrites live.html ATOMICALLY (tmp + os.replace) so the
      auto-refresh can never read a half-written file.

Design rules honoured:
    * Self-contained: inline SVG + inline CSS, NO external/CDN assets -> file://
      works fully offline.
    * Import-light + crash-proof: `import chess.svg` is lazy and EVERY public
      method swallows its own exceptions. A viz failure must NEVER take down the
      game loop -- the terminal SAN line keeps working regardless.

Public API:
    viz = LiveViz(out_dir)
    viz.start(title="...")                     # write initial page + open browser once
    viz.update(board, move_list, header, curve_points)   # rewrite live.html
    path = viz.html_path                        # absolute path to live.html

`curve_points` is a list of dicts as written by train_robust's strength_curve.json
(keys used: 'iter', 'winrate_vs_random', 'winrate_vs_classical_d1'); any subset is
tolerated and missing keys are skipped.
"""
from __future__ import annotations

import html as _htmllib
import os
import time
import webbrowser
from typing import List, Optional, Sequence


class LiveViz:
    """Live self-contained HTML board writer (no server, no CDN, atomic writes)."""

    def __init__(self, out_dir: str):
        self.out_dir = os.path.abspath(out_dir)
        self.html_path = os.path.join(self.out_dir, "live.html")
        self._opened = False
        self._ok = True            # flips False on a hard failure -> degrade silently
        self._title = "chess_zero -- learn-watch live"
        try:
            os.makedirs(self.out_dir, exist_ok=True)
        except Exception:
            self._ok = False

    # ----------------------------------------------------------------- start --
    def start(self, title: Optional[str] = None, open_browser: bool = True) -> None:
        """Write an initial auto-refreshing page and open it ONCE in the browser.

        Safe to call once at the top of the watch loop. Never raises."""
        if not self._ok:
            return
        if title:
            self._title = title
        try:
            placeholder = (
                '<div class="wait">Waiting for the first move&hellip;</div>')
            self._write_html(self._render(
                board_svg=placeholder, move_html="", header=title or self._title,
                curve_points=[]))
        except Exception:
            self._ok = False
            return
        if open_browser and not self._opened:
            try:
                webbrowser.open("file://" + self.html_path.replace(os.sep, "/"))
            except Exception:
                pass  # headless / no browser -> the file is still valid on disk
            self._opened = True

    # ---------------------------------------------------------------- update --
    def update(self, board, move_list: Sequence[str], header: str,
               curve_points: Optional[List[dict]] = None) -> None:
        """Rewrite live.html for the current position. Never raises.

        board        : a chess.Board (current position; last move highlighted)
        move_list    : list of SAN strings played so far
        header       : matchup + which iter the live net is from
        curve_points : list of strength_curve rows (see module docstring)
        """
        if not self._ok:
            return
        try:
            board_svg = self._board_svg(board)
            move_html = self._move_list_html(move_list or [])
            html = self._render(board_svg=board_svg, move_html=move_html,
                                header=header, curve_points=curve_points or [])
            self._write_html(html)
        except Exception:
            # A single bad frame must not stop the game; keep the loop alive.
            return

    # -------------------------------------------------------------- internals --
    def _board_svg(self, board) -> str:
        """Real-piece SVG via python-chess (lazy import; degrade to a note)."""
        try:
            import chess.svg  # lazy: only imported when we actually render
            last = board.peek() if board.move_stack else None
            return chess.svg.board(board, lastmove=last, size=480)
        except Exception:
            return ('<div class="wait">(board SVG unavailable -- see terminal '
                    'SAN line)</div>')

    @staticmethod
    def _move_list_html(move_list: Sequence[str]) -> str:
        """SAN move list as '1. e4 e5  2. Nf3 ...' with paired numbering."""
        if not move_list:
            return '<span class="muted">(no moves yet)</span>'
        cells = []
        for i, san in enumerate(move_list):
            esc = _htmllib.escape(str(san))
            if i % 2 == 0:
                cells.append('<span class="mn">%d.</span> <span class="san">%s</span>'
                             % (i // 2 + 1, esc))
            else:
                cells.append('<span class="san">%s</span>' % esc)
        return " ".join(cells)

    @staticmethod
    def _curve_svg_and_table(curve_points: List[dict]) -> str:
        """Inline-SVG sparkline (THREE series) + a compact recent-iters table.

        Series, y in [0,1]:
          * winrate_vs_random      (green)  -- the monotonic floor vs a random mover.
          * winrate_vs_classical_d1(orange) -- wins-only rate vs the classical engine;
            this sits PINNED at 0.0 early (the net loses outright), so it is the FLAT line.
          * score_vs_classical_d1  (blue)   -- the DRAW-AWARE climb axis (wins+0.5*draws)/g.
            Progress vs the engine first shows up as losing -> DRAWING, which leaves the
            orange win-rate flat at 0.0 but RAISES this blue score from 0 toward 0.5. This
            is the line the curriculum/climb gate actually moves, so it is the one to watch.
        Pure SVG/HTML -- no JS, no CDN."""
        # Collect clean (iter, vr, vc, vs) rows. vs = the draw-aware climb score.
        rows = []
        for cp in curve_points:
            if not isinstance(cp, dict):
                continue
            it = cp.get("iter")
            vr = cp.get("winrate_vs_random")
            vc = cp.get("winrate_vs_classical_d1")
            vs = cp.get("score_vs_classical_d1")
            if it is None:
                continue
            rows.append((it, vr, vc, vs))
        if not rows:
            return ('<div class="muted">strength curve: (no eval points yet -- '
                    'the trainer fills this in as iters complete)</div>')

        W, H, pad = 360, 110, 8
        its = [r[0] for r in rows]
        imin, imax = min(its), max(its)
        span = (imax - imin) or 1

        def _x(it):
            return pad + (it - imin) / span * (W - 2 * pad)

        def _y(v):
            v = max(0.0, min(1.0, float(v)))
            return H - pad - v * (H - 2 * pad)

        def _poly(series_idx):
            pts = []
            for it, vr, vc, vs in rows:
                v = (vr if series_idx == 0 else vc if series_idx == 1 else vs)
                if v is None:
                    continue
                pts.append("%.1f,%.1f" % (_x(it), _y(v)))
            return " ".join(pts)

        poly_r = _poly(0)
        poly_c = _poly(1)
        poly_s = _poly(2)  # the draw-aware CLIMB axis
        # gridlines at 0 / 0.5 / 1.0
        grid = ""
        for gv in (0.0, 0.5, 1.0):
            gy = _y(gv)
            grid += ('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" '
                     'stroke="#333a4d" stroke-width="1"/>'
                     '<text x="2" y="%.1f" fill="#9aa3b8" font-size="8">%.1f</text>'
                     % (pad, gy, W - pad, gy, gy + 3, gv))
        line_r = ('<polyline fill="none" stroke="#36c08a" stroke-width="2" '
                  'points="%s"/>' % poly_r) if poly_r else ""
        line_c = ('<polyline fill="none" stroke="#f0a93b" stroke-width="2" '
                  'points="%s"/>' % poly_c) if poly_c else ""
        # blue climb axis drawn LAST (on top) so the watchable line is never hidden.
        line_s = ('<polyline fill="none" stroke="#4d9bf0" stroke-width="2" '
                  'points="%s"/>' % poly_s) if poly_s else ""
        svg = ('<svg viewBox="0 0 %d %d" width="100%%" height="%d" '
               'preserveAspectRatio="none" style="background:#1a1d29;'
               'border:1px solid #333a4d;border-radius:8px">%s%s%s%s</svg>'
               % (W, H, H, grid, line_r, line_c, line_s))

        # Compact table of the last few rows.
        tail = rows[-6:]
        trs = ""
        for it, vr, vc, vs in tail:
            vr_s = "%.3f" % vr if vr is not None else "&middot;"
            vc_s = "%.3f" % vc if vc is not None else "&middot;"
            vs_s = "%.3f" % vs if vs is not None else "&middot;"
            trs += ('<tr><td>%s</td><td class="g">%s</td><td class="o">%s</td>'
                    '<td class="b">%s</td></tr>'
                    % (it, vr_s, vc_s, vs_s))
        table = (
            '<table class="curve"><thead><tr><th>iter</th>'
            '<th class="g">vs random</th><th class="o">vs classical</th>'
            '<th class="b">climb score</th></tr>'
            '</thead><tbody>%s</tbody></table>' % trs)

        legend = ('<div class="legend">'
                  '<span class="dot g"></span> winrate vs random &nbsp;&nbsp;'
                  '<span class="dot o"></span> winrate vs classical &nbsp;&nbsp;'
                  '<span class="dot b"></span> climb score (draw-aware)</div>')
        return legend + svg + table

    # ---------------------------------------------------------------- render --
    def _render(self, board_svg: str, move_html: str, header: str,
                curve_points: List[dict]) -> str:
        curve_html = self._curve_svg_and_table(curve_points or [])
        head = _htmllib.escape(str(header or self._title))
        title = _htmllib.escape(self._title)
        ts = time.strftime("%H:%M:%S")
        # NOTE: meta-refresh keeps the page live with NO server. Everything inline.
        return _PAGE % {
            "title": title,
            "header": head,
            "board_svg": board_svg,     # raw SVG -- trusted (we generate it)
            "move_html": move_html,
            "curve_html": curve_html,
            "ts": ts,
        }

    def _write_html(self, html: str) -> None:
        """Atomic write: tmp + os.replace so the meta-refresh never reads a
        half-written file."""
        tmp = self.html_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(html)
        os.replace(tmp, self.html_path)


# Self-contained page template (inline CSS, inline SVG, meta-refresh; no CDN).
# %(...)s tokens are filled by _render; literal CSS braces are fine (str %, not f-string).
_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="1">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%(title)s</title>
<style>
  :root { --bg:#0f1117; --panel:#1a1d29; --panel2:#232735; --ink:#e8eaf0;
          --muted:#9aa3b8; --good:#36c08a; --accent:#f0a93b; --climb:#4d9bf0;
          --line:#333a4d; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         line-height:1.45; }
  header { padding:16px 22px; background:var(--panel); border-bottom:1px solid var(--line); }
  header h1 { margin:0 0 4px; font-size:18px; }
  header .matchup { color:var(--accent); font-size:14px; font-weight:600; }
  header .ts { color:var(--muted); font-size:11px; margin-top:4px; }
  .wrap { display:flex; gap:22px; padding:22px; flex-wrap:wrap; align-items:flex-start; }
  .col-board { flex:0 0 auto; }
  .board { background:var(--panel); border:1px solid var(--line); border-radius:10px;
           padding:14px; display:inline-block; }
  .board svg { display:block; }
  .wait { color:var(--muted); width:480px; height:480px; display:flex;
          align-items:center; justify-content:center; font-size:15px; }
  .col-side { flex:1 1 380px; min-width:340px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px;
          padding:14px 16px; margin-bottom:16px; }
  .card h3 { margin:0 0 10px; font-size:14px; }
  .moves { font-size:14px; line-height:1.9; word-spacing:2px; }
  .moves .mn { color:var(--muted); margin-left:8px; }
  .moves .san { font-weight:600; }
  .muted { color:var(--muted); font-size:13px; }
  .legend { font-size:12px; color:var(--muted); margin-bottom:8px; }
  .dot { display:inline-block; width:10px; height:10px; border-radius:50%%;
         vertical-align:middle; margin-right:3px; }
  .dot.g { background:var(--good); } .dot.o { background:var(--accent); }
  .dot.b { background:var(--climb); }
  table.curve { width:100%%; border-collapse:collapse; margin-top:10px; font-size:12.5px; }
  table.curve th, table.curve td { text-align:right; padding:3px 8px;
         border-bottom:1px solid var(--line); }
  table.curve th:first-child, table.curve td:first-child { text-align:left; }
  table.curve td.g, table.curve th.g { color:var(--good); }
  table.curve td.o, table.curve th.o { color:var(--accent); }
  table.curve td.b, table.curve th.b { color:var(--climb); }
  .foot { padding:6px 22px 22px; color:var(--muted); font-size:11.5px; }
</style>
</head>
<body>
<header>
  <h1>chess_zero &mdash; learn-watch (live)</h1>
  <div class="matchup">%(header)s</div>
  <div class="ts">live board &mdash; auto-refreshes every 1s &middot; updated %(ts)s</div>
</header>
<div class="wrap">
  <div class="col-board">
    <div class="board">%(board_svg)s</div>
  </div>
  <div class="col-side">
    <div class="card">
      <h3>Moves (SAN)</h3>
      <div class="moves">%(move_html)s</div>
    </div>
    <div class="card">
      <h3>Strength curve</h3>
      %(curve_html)s
    </div>
  </div>
</div>
<div class="foot">
  Self-contained: the board is inline SVG from <code>chess.svg.board()</code> (real
  pieces, no CDN); the curve is inline SVG/HTML; the page meta-refreshes (no server).
</div>
</body>
</html>
"""
