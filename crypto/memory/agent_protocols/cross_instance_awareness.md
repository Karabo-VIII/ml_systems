# Cross-Instance Awareness Protocol

> Codifies the MEMORY.md feedback line that says "delta-check
> `~/.claude/projects/*.jsonl` mtimes + `git log --since='1 hour ago'`".
> Failing to do this caused today's "frontier orphaned" mis-claim (a 2026-05-12
> wire happened the day before my 2026-05-13 synthesis).

## Trigger

- **Session start** — always
- **Every ~30 min** of active work
- **Before any high-stakes claim about project state** (especially after Sonnet scout reports)

## Steps

### 1. Git log check (1 command)
```bash
git log --since="1 hour ago" --pretty=format:"%h %ad %s" --date=short
```

If output is non-empty: another Claude instance OR human has committed work
since your last check. **Read each commit's diff before proceeding.**

### 2. Worktree status check
```bash
git status -s | head -20
```

If files were modified outside this session, those modifications could
contradict your assumptions. Read the modified files before claiming state.

### 3. Other-session jsonl timestamps (cross-session awareness)
Check `~/.claude/projects/<slug>/*.jsonl` mtimes (Windows: `%USERPROFILE%`).
A fresh mtime within last hour = another instance is working.

### 4. Update mental model BEFORE claiming state
Anything you "know" about project state must be invalidated against the
results of steps 1-3. Specifically:
- "X is orphaned" — re-verify by reading current files
- "Y is wired" — re-verify with the actual config / runner
- "Last session shipped Z" — verify via `git log` not memory

## Failure mode this prevents

**2026-05-13 today**: my `META_ROI_SYNTHESIS_2026_05_13.md` claimed
`v6_frontier` and `v7_frontier` were ORPHANED. Source: memory memo dated
2026-05-11. Reality: commit `2345a48` on 2026-05-12 added the
`V6_FRONTIER_v2026_05` blend. My synthesis ran **one day stale**.

Cost: user had to point out the rewire was already done. Re-work loss: 
~10 minutes of confusion + the inventory pivot.

Had I run `git log --since="1 day ago"` at synthesis-start, I'd have seen the
commit and avoided the wrong claim entirely.

## Anti-pattern: "I just loaded MEMORY.md, that's enough"

MEMORY.md is updated when memories are written. Code changes happen between
memory writes. A 1-day-old memory is older than a 1-hour-old commit. **Always
re-check git log before claiming code state, regardless of MEMORY.md
recency.**

## Quick boilerplate (run at session start)

```bash
echo "=== Cross-Instance Awareness Check ==="
echo "git log last 24h:"
git log --since="24 hours ago" --pretty=format:"  %h %ad %s" --date=short | head -10
echo
echo "uncommitted:"
git status -s | head -10
echo
echo "memory mtimes (top 5 most recent):"
ls -lt memory/*.md 2>&1 | head -5
```

If any of these show recent activity → read the relevant artifacts before
proceeding.

## Integration with sessions

This protocol's run is logged in the user-facing report:
> "Cross-instance check: 2 commits in last 24h (`2345a48` V6_FRONTIER wire,
> `915a744` Round 7); 0 uncommitted files; memory current. Proceeding."

If the check surfaces a relevant change, the user-facing report explicitly
acknowledges it:
> "Cross-instance check: detected `V6_FRONTIER_v2026_05` was wired yesterday
> (commit 2345a48). Updating my mental model — frontier is NOT orphaned; my
> earlier 2026-05-11 memo is stale on this point."
