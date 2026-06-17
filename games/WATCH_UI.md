# Watch the engines play -- live browser UI

Every engine now has a real **graphical** view (not just the ASCII terminal) -- the same
self-contained, no-server approach the chess learn-watch loop already used. Add `--web` to
any run command: it writes a live, auto-refreshing page to `runs/viz/live.html` and opens it
in your default browser. Keep the **one** tab open -- each engine rewrites it as it plays.

## Commands (run these locally so the browser opens for you)

```
# all three engines, back to back, in the browser
python run_engines.py --web

# one engine at a time
python play_chess.py --web                 # AlphaZero champion self-play (real piece graphics)
python play_connect4.py --web              # red/yellow discs drop into the board
python play_atari.py --web                 # real MinAtar Breakout (paddle/ball/bricks as cells)
python play_atari.py --mode catch --web    # the MuZero/Catch variant (planning over a learned model)

# pacing + scope
python run_engines.py --web --delay 0.6    # slower, easier to watch
python play_atari.py --game all --web      # all 3 MinAtar games (breakout/space_invaders/asterix)
python run_engines.py --web --no-render    # browser only (no ASCII in the terminal)
```

## What you see
- **Chess** -- the real board (python-chess SVG), last move highlighted, the SAN move list.
- **Connect-4** -- a blue 6x7 board; red = player 0 (the net when it plays X), yellow = player 1.
- **Atari / MinAtar** -- the 10x10 game grid, one color per channel (paddle / ball / trail / bricks),
  score in the header.
- **Catch (MuZero)** -- the 5x5 grid, cyan ball + green paddle.

## How it works (no setup, no server)
- `--web` builds a self-contained `runs/viz/live.html` (inline SVG/CSS, no CDN) and opens it once.
- The page reloads itself (JS every ~0.45s, with a `<meta http-equiv="refresh">` fallback), so it
  stays a LIVE view with **no server**.
- Writes are atomic (tmp + replace) and the visualizer is crash-proof -- a viz hiccup never stops
  the game (the terminal render keeps working regardless).
- When a game ends the page settles on the final frame (a FINAL badge) and stops reloading.
- `runs/` is gitignored. The module is `az/web_viz.py`; `--web` is on `run_engines.py` and every
  `play_*.py`.

Notes:
- `--web` keeps the terminal render ON too (that is what paces the frames); use `--delay` to slow it.
- Headless / no browser wanted? set `GAMESENGINE_NO_BROWSER=1` and just open `runs/viz/live.html`
  yourself -- the file is always written either way.
