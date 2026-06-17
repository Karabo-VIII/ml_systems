"""
az.web_viz -- a TRUE browser visualizer for ALL the demo engines (not just chess).

The terminal ASCII grid is not a real visualizer. This writes a self-contained HTML
page (out_dir/live.html) that shows the live game with real graphics:
  * GRID games (Connect-4, MinAtar Breakout/Space-Invaders/Asterix, Catch) -> a CSS
    grid of coloured cells, with a legend.
  * CHESS -> real piece graphics via python-chess `chess.svg.board()` (inline SVG).

It mirrors the chess `live_viz.LiveViz` design, generalised to any engine:

How it stays a LIVE view with NO server:
    * the page reloads itself (a JS `setTimeout(location.reload, ~450ms)` for a smooth
      sub-second update, plus a `<meta http-equiv="refresh">` fallback if JS is off);
    * every update rewrites live.html ATOMICALLY (tmp + os.replace) so a reload can
      never read a half-written file;
    * when the game ends, `done=True` drops the auto-reload so the page settles on the
      final frame (with a FINAL badge).

Design rules honoured (same as live_viz):
    * Self-contained: inline CSS + inline SVG, NO external/CDN assets -> file:// works
      fully offline.
    * Crash-proof: EVERY public method swallows its own exceptions. A viz failure must
      NEVER take down the game loop -- the terminal render keeps working regardless.

Public API:
    viz = LiveGameViz(out_dir, title="Connect-4 -- AlphaZero", subtitle="...")
    viz.start(open_browser=True)                         # initial page + open browser once
    viz.grid(cells, palette, header=..., status=..., legend=..., done=False)
    viz.board_svg(svg, header=..., moves_html=..., status=..., done=False)
    path = viz.html_path                                 # absolute path to live.html

`cells` is a 2D list (rows x cols) of ints; each int indexes `palette` (index 0 = empty).
`palette` is a list of (label, css_color) pairs; entry 0 is the empty/background cell.
No emoji / non-ASCII (Windows cp1252 safety).
"""
from __future__ import annotations

import html as _htmllib
import os
import time
import webbrowser
from typing import List, Optional, Sequence, Tuple


class LiveGameViz:
    """Live self-contained HTML game writer (no server, no CDN, atomic writes, crash-proof)."""

    def __init__(self, out_dir: str, title: str = "games_engine -- live",
                 subtitle: str = "", reload_ms: int = 450):
        self.out_dir = os.path.abspath(out_dir)
        self.html_path = os.path.join(self.out_dir, "live.html")
        self.title = title
        self.subtitle = subtitle
        self.reload_ms = max(150, int(reload_ms))
        self._opened = False
        self._ok = True
        try:
            os.makedirs(self.out_dir, exist_ok=True)
        except Exception:
            self._ok = False

    # ----------------------------------------------------------------- start --
    def start(self, open_browser: bool = True) -> None:
        """Write an initial 'waiting' page and open it ONCE in the browser. Never raises."""
        if not self._ok:
            return
        try:
            body = '<div class="wait">Waiting for the first frame&hellip;</div>'
            self._write(self._page(body=body, header=self.title, status="", done=False))
        except Exception:
            self._ok = False
            return
        if open_browser and not self._opened:
            try:
                webbrowser.open("file://" + self.html_path.replace(os.sep, "/"))
            except Exception:
                pass  # headless / no browser -> the file is still valid on disk
            self._opened = True

    # ------------------------------------------------------------------ grid --
    def grid(self, cells: Sequence[Sequence[int]],
             palette: Sequence[Tuple[str, str]],
             header: str = "", status: str = "",
             legend: bool = True, board_bg: str = "#10131c",
             shape: str = "round", done: bool = False,
             col_labels: Optional[Sequence] = None) -> None:
        """Render a rows x cols grid of coloured cells. `cells[r][c]` indexes `palette`.

        palette[0] is the EMPTY cell; palette[k>0] is an entity (label, css_color).
        shape: 'round' (discs / dots) or 'square'. col_labels: optional per-column captions
        (e.g. Connect-4 column numbers) shown under the grid. Never raises."""
        if not self._ok:
            return
        try:
            body = self._grid_html(cells, palette, board_bg, shape, legend, col_labels)
            self._write(self._page(body=body, header=header or self.title,
                                    status=status, done=done))
        except Exception:
            return

    # ------------------------------------------------------------ board_svg --
    def board_svg(self, svg: str, header: str = "", moves_html: str = "",
                  status: str = "", done: bool = False) -> None:
        """Render a chess board from a raw SVG string (e.g. chess.svg.board(...)). Never raises."""
        if not self._ok:
            return
        try:
            side = ""
            if moves_html:
                side = ('<div class="moves"><h3>Moves</h3><div class="movelist">%s</div></div>'
                        % moves_html)
            body = '<div class="boardwrap"><div class="board">%s</div>%s</div>' % (svg, side)
            self._write(self._page(body=body, header=header or self.title,
                                    status=status, done=done))
        except Exception:
            return

    # -------------------------------------------------------------- internals --
    @staticmethod
    def _grid_html(cells, palette, board_bg, shape, legend, col_labels=None) -> str:
        rows = list(cells)
        ncols = max((len(r) for r in rows), default=0)
        radius = "50%" if shape == "round" else "16%"
        out = ['<div class="gridwrap" style="background:%s">' % board_bg]
        out.append('<div class="grid" style="grid-template-columns:repeat(%d,var(--cell))">' % ncols)
        for row in rows:
            for v in row:
                v = int(v) if v is not None else 0
                if v <= 0 or v >= len(palette):
                    out.append('<div class="cell empty" style="border-radius:%s"></div>' % radius)
                else:
                    label, color = palette[v]
                    out.append('<div class="cell disc" style="background:%s;border-radius:%s" title="%s"></div>'
                               % (color, radius, _htmllib.escape(str(label))))
        out.append('</div>')  # .grid
        if col_labels:
            out.append('<div class="collabels" style="grid-template-columns:repeat(%d,var(--cell))">' % ncols)
            for lab in list(col_labels)[:ncols]:
                out.append('<div class="collabel">%s</div>' % _htmllib.escape(str(lab)))
            out.append('</div>')
        out.append('</div>')  # .gridwrap
        if legend and len(palette) > 1:
            sw = []
            for label, color in palette[1:]:
                sw.append('<span class="lg"><span class="sw" style="background:%s"></span>%s</span>'
                          % (color, _htmllib.escape(str(label))))
            out.append('<div class="legend">' + " ".join(sw) + '</div>')
        return "".join(out)

    def _page(self, body: str, header: str, status: str, done: bool) -> str:
        ts = time.strftime("%H:%M:%S")
        head = _htmllib.escape(str(header or self.title))
        sub = _htmllib.escape(str(self.subtitle))
        stat = _htmllib.escape(str(status or ""))
        title = _htmllib.escape(str(self.title))
        # The page ALWAYS keeps reloading -- fast while a game is live, slow once it ends --
        # so a finished game NEVER hard-freezes. The same tab then keeps showing the next
        # engine in a combined run, or the next run you start. (A hard stop here was the
        # 'stuck after a while' bug: the first engine to finish froze the tab.)
        interval = max(self.reload_ms * 4, 1800) if done else self.reload_ms
        if done:
            badge = ('<span class="badge done">GAME OVER</span> '
                     'waiting for the next game/run &middot; %s' % ts)
        else:
            badge = '<span class="badge live">LIVE</span> auto-updates &middot; %s' % ts
        reload_head = ('<meta http-equiv="refresh" content="%d">'
                       '<script>setTimeout(function(){location.reload();},%d);</script>'
                       % (max(2, round(interval / 1000.0)), interval))
        sub_html = ('<div class="sub">%s</div>' % sub) if sub else ""
        stat_html = ('<div class="status">%s</div>' % stat) if stat else ""
        return (_HEAD_A + reload_head + _HEAD_B
                + ("<title>%s</title>" % title) + _CSS + _HEAD_C
                + ('<header><h1>%s</h1>%s<div class="meta">%s</div>%s</header>'
                   % (head, sub_html, badge, stat_html))
                + ('<main>%s</main>' % body)
                + _FOOT)

    def _write(self, html: str) -> None:
        tmp = self.html_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(html)
        os.replace(tmp, self.html_path)


# --------------------------------------------------------------------------- #
# Page chrome (split so the optional reload <meta>/<script> can be injected in <head>).
# Inline CSS only -- no CDN. CSS braces are literal (plain string, not %/format).
# --------------------------------------------------------------------------- #
_HEAD_A = ('<!DOCTYPE html>\n<html lang="en">\n<head>\n'
           '<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1">\n')
_HEAD_B = ""
_HEAD_C = "\n</head>\n<body>\n"
_FOOT = ('\n<footer>Self-contained live view (inline SVG/CSS, no server, no CDN) &middot; '
         'games_engine</footer>\n</body>\n</html>\n')

_CSS = """<style>
  :root { --bg:#0f1117; --panel:#1a1d29; --ink:#e8eaf0; --muted:#9aa3b8;
          --line:#333a4d; --accent:#f0a93b; --cell:clamp(26px,8.5vmin,60px); }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); line-height:1.45;
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
  header { padding:16px 22px; background:var(--panel); border-bottom:1px solid var(--line); }
  header h1 { margin:0 0 4px; font-size:18px; }
  header .sub { color:var(--accent); font-size:13.5px; font-weight:600; margin-bottom:4px; }
  header .meta { color:var(--muted); font-size:11.5px; }
  header .status { color:var(--ink); font-size:13px; margin-top:6px; font-variant-numeric:tabular-nums; }
  .badge { font-size:10.5px; font-weight:700; padding:1px 7px; border-radius:9px; vertical-align:middle; }
  .badge.live { background:#10331f; color:#36c08a; border:1px solid #1c5b3a; }
  .badge.done { background:#3a2a10; color:#f0a93b; border:1px solid #6b4d1c; }
  main { padding:24px 22px; }
  .wait { color:var(--muted); padding:60px; text-align:center; font-size:15px; }
  .gridwrap { display:inline-block; padding:14px; border-radius:14px; border:1px solid var(--line); }
  .grid { display:grid; gap:6px; }
  .cell { width:var(--cell); height:var(--cell); }
  .cell.empty { background:rgba(0,0,0,0.34); box-shadow:inset 0 2px 6px rgba(0,0,0,0.55); }
  .cell.disc { box-shadow:inset 0 -3px 7px rgba(0,0,0,0.33), inset 0 3px 6px rgba(255,255,255,0.28); }
  .collabels { display:grid; gap:6px; margin-top:8px; }
  .collabel { text-align:center; color:var(--muted); font-size:12.5px; font-variant-numeric:tabular-nums; }
  .legend { margin-top:14px; color:var(--muted); font-size:12.5px; }
  .legend .lg { margin-right:14px; white-space:nowrap; }
  .legend .sw { display:inline-block; width:12px; height:12px; border-radius:3px;
                vertical-align:middle; margin-right:5px; }
  .boardwrap { display:flex; gap:22px; flex-wrap:wrap; align-items:flex-start; }
  .board { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px; }
  .board svg { display:block; }
  .moves { flex:1 1 280px; min-width:240px; background:var(--panel);
           border:1px solid var(--line); border-radius:12px; padding:12px 16px; }
  .moves h3 { margin:0 0 8px; font-size:14px; }
  .movelist { font-size:13.5px; line-height:1.9; word-spacing:2px; }
  .movelist .mn { color:var(--muted); margin-left:8px; }
  footer { padding:10px 22px 22px; color:var(--muted); font-size:11px; }
</style>"""


def moves_to_html(san_list: Sequence[str]) -> str:
    """Format a list of SAN strings as '1. e4 e5  2. Nf3 ...' for board_svg(moves_html=...)."""
    cells = []
    for i, san in enumerate(san_list or []):
        esc = _htmllib.escape(str(san))
        if i % 2 == 0:
            cells.append('<span class="mn">%d.</span> %s' % (i // 2 + 1, esc))
        else:
            cells.append(esc)
    return " ".join(cells) if cells else '<span class="mn">(no moves yet)</span>'
