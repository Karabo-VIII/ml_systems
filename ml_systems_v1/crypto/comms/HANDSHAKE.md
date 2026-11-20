# Handshake — operational specifics

Fills the gaps between PROTOCOL.md's concepts and the actual filesystem / JSONL operations. Read this before your first turn in any session.

## 1. Finding your OWN JSONL (self-discovery)

Problem: when two Claude sessions are active, "most recently modified JSONL" is ambiguous.

**Solution: handshake-token grep.**

In your FIRST reply (bootstrap), include a distinctive, unique phrase in your chat output:

```
HANDSHAKE_TOKEN: <random-uuid-or-unique-phrase>
```

Then run:
```bash
grep -l "HANDSHAKE_TOKEN: <your-token>" ~/.claude/projects/<slug>/*.jsonl
```

The returned file is your own JSONL. Record that path in your bootstrap turn marker + session.yaml.

Concrete example (Windows Bash):
```bash
grep -l "HANDSHAKE_TOKEN: 7f3a2b8c" C:/Users/karab/.claude/projects/c--Users-karab-Documents-coding-v4-crypto-stystem/*.jsonl
```

**Alternative:** the harness may expose `$CLAUDE_SESSION_FILE` or similar env var. Check `env | grep -i claude` at session start. If available, use directly.

## 2. Reading the OTHER agent's JSONL slice

Claude Code stores transcripts as JSONL. Each line is one turn (user prompt or assistant reply or tool call).

**To read only the other agent's most recent reply:**

```python
import json
with open(other_jsonl, 'r', encoding='utf-8') as f:
    entries = [json.loads(line) for line in f if line.strip()]
# Filter to assistant messages after my last turn timestamp
my_last_turn = "2026-04-24T15:02:10Z"
replies = [e for e in entries if
           e.get("type") == "assistant" and
           e.get("timestamp", "") >= my_last_turn]
# The last one in `replies` is their latest reply
```

**Shortcut for human-readable scan:**
```bash
tail -100 "<other-jsonl>" | grep '"type":"assistant"' | tail -3
```

**Content extraction:** the reply text lives at `message.content[*].text` for text blocks. Tool-use blocks are at `message.content[*].input`. Read both — the reply MAY include tool calls + their results.

### 2.1 Side-channel: human-to-other-agent messages (REQUIRED)

Human side-channel discussions happen between "official" turn boundaries. Example flow:
1. Alpha drops turn marker `005` (official peer-turn)
2. Human reviews and injects clarifications to Alpha in chat
3. Alpha revises reply + updates marker to `005 v2`
4. Bravo is activated

If Bravo only reads Alpha's `005 v2` assistant reply, they miss the HUMAN's clarifying messages that triggered the revision. Those messages are binding project context.

**VERIFY step must ALSO read `user`-role entries in the other agent's JSONL that occurred since the last peer-exchange.**

```python
# Human-injected messages since peer's last reply to ME
last_peer_assistant_ts = "2026-04-24T09:34:12Z"   # last time peer replied to me
human_msgs = [e for e in entries if
              e.get("type") == "user" and
              e.get("timestamp", "") >= last_peer_assistant_ts and
              # exclude system reminders; focus on user.content[*].text
              any(c.get("type") == "text" for c in e.get("message", {}).get("content", []))]
```

The assistant-reply-preceding-a-user-msg-followed-by-a-second-assistant-reply pattern = HUMAN SIDE-CHANNEL. Read both assistant replies AND the human msgs in between.

**Signal in the turn marker** that you processed these:
```yaml
verifications_run:
  - "read peer's assistant JSONL slice [N messages]"
  - "read peer's USER side-channel [M human-injected directives since last peer-reply]"
```

And the writing agent MUST record side-channel directives they applied in their own turn marker:
```yaml
human_directives_received:
  - "concrete directive 1 (e.g., 'limit non-CEX infra')"
  - "concrete directive 2"
```

This closes the loop: reader knows where to look, writer has flagged what was absorbed.

## 3. Bootstrap ordering — first-writer-wins

To avoid the race where both chats try to create `session.yaml`:

**Rule:** the agent whose name is alphabetically FIRST creates the session directory + yaml. The other appends themselves to `agents:` list.

Alpha vs Bravo → Alpha creates. If names don't sort obviously (e.g. Maker vs Breaker), human declares creator in the bootstrap prompt.

**If both write anyway (race):**
- `git status` shows conflict on session.yaml
- Resolve by keeping the first-committed version (by mtime); the other appends via a second commit

## 4. Session.yaml mutation rules

Who can write `session.yaml`:
- **Creator**: at bootstrap, writes the initial file
- **Joiner**: appends themselves to `agents:` list, nothing else
- **Each turn**: only the agent whose turn just completed may update `current_turn` and `next_up`
- **Constitution changes**: require a dedicated DEBATE thread + CONCEDE from all agents; the agent with the last concession commits the change

## 5. Cross-instance directive integration

The project's `memory/MEMORY.md` has a cross-instance awareness directive:
> "session start + every ~30min, delta-check `~/.claude/projects/<slug>/*.jsonl` mtimes + `git log --since='1 hour ago'` + `git status`. Concurrent = collaboration."

**Inside a comms session:** the other collaborator in `session.yaml` is the PRIMARY peer. Their updates are via turn markers, not via mtime scanning.

**Outside a comms session:** if during VERIFY an agent discovers a JSONL they don't recognize (not in `session.yaml`), that's a THIRD-PARTY instance. Protocol:
- Note it in your turn marker under `external_context_seen:` (list of unknown jsonls + git commits since last turn)
- Don't assume they're collaborating on your session
- If their work conflicts with current session plan, REVIEW or file BLOCK with reference
- Alternative: ask human to add them to session as another collaborator

## 6. VERIFY token budget

Don't read the entire other-agent JSONL every turn. Apply these rules:

- **Minimal**: just their latest assistant message (post `reply_marker` timestamp of most recent turn marker) + any `user`-role side-channel messages since their last reply to you (see §2.1)
- **Default**: last 3 turns of their JSONL (for conversational context) + all side-channel user msgs in the same window
- **Expanded**: up to 10 turns if they reference older context or you're reviewing a long-running thread
- **Full**: only if you're joining a session mid-flight (catch-up)

Side-channel (human-to-peer) messages are ALWAYS included at every budget level. Even "minimal" must read them — they can carry binding constraints.

Log which you did in `verifications_run:` so the other agent knows how much of their context you read.

## 7. Work-in-progress (WIP) signal

If your turn is a partial progress report (not end of a task):

```yaml
---
sub_protocol: REPORT
status: wip                              # NEW: wip | requires_response | terminal
progress_pct: 60
blocking_on: none                        # or "Bravo to clarify spec X"
---
```

Other agent's response should generally be patience (INFORMATIONAL ack) unless you're `blocking_on: them`.

## 8. Artifact existence check (VERIFY-time)

Before trusting claims about artifacts, check them:
```bash
# File exists?
ls -la <artifact_path> 2>&1

# Committed yet?
git log --oneline -5 -- <artifact_path>

# Or uncommitted changes?
git diff HEAD -- <artifact_path> | head -30
```

If a referenced artifact doesn't exist or is uncommitted, note in your REVIEW: "your turn marker references `scripts/X.py` but I can't find it / it has uncommitted changes." Don't make up what would be there.

## 9. Bootstrap JSONL path exchange

Chicken-and-egg: Alpha doesn't know Bravo's path until Bravo joins. Handle like this:

### Step A — Alpha's bootstrap (no Bravo yet)
Alpha creates session.yaml with:
```yaml
agents:
  - name: Alpha
    jsonl_path: "<discovered via handshake-token grep>"
  - name: Bravo
    jsonl_path: "<pending>"
    status: not_joined
```

Alpha's turn marker `to:` field uses `Bravo (pending)`.

### Step B — Bravo joins
Bravo reads session.yaml, runs own handshake-token + grep to find own JSONL, fills it in. Alpha's path was already listed; Bravo confirms by reading Alpha's first turn marker + JSONL.

```yaml
agents:
  - name: Alpha
    jsonl_path: "C:/Users/karab/.claude/projects/.../{alpha-uuid}.jsonl"
  - name: Bravo
    jsonl_path: "C:/Users/karab/.claude/projects/.../{bravo-uuid}.jsonl"
    status: joined
```

### Step C — Alpha's next turn
Alpha now sees Bravo's actual path + first turn marker. Can VERIFY properly. Session is fully bonded.

## 10. Archival policy

When a session ends (`status: ended` via DECISION turn marker):
- Leave `comms/SESSIONS/<id>/` in place. Don't move.
- Optionally create `comms/SESSIONS/<id>/SUMMARY.md` — a one-page distillation of key outcomes
- `git log` + the turn markers + JSONLs preserve the full trace

## 11. Quick reference — what each agent does at bootstrap

### Both agents, first turn
1. Generate unique handshake token
2. Output `HANDSHAKE_TOKEN: <token>` in your first chat message
3. After your first reply lands in YOUR jsonl, run `grep -l <token> <claude-projects-dir>/*.jsonl` to discover own path
4. Read `comms/PROTOCOL.md`, `comms/ROLES.md`, `comms/HANDSHAKE.md`
5. Read `comms/SESSIONS/<id>/` if it exists

### Alpha (creator) only
6. Create `comms/SESSIONS/<id>/session.yaml` with:
   - yourself fully filled
   - placeholder for Bravo (`<pending>`)
7. Drop `turns/001_alpha_PROPOSE.md` announcing session + initial proposal
8. Your reply body includes HANDSHAKE_TOKEN + announces you're waiting for Bravo

### Bravo (joiner) only
6. Read `session.yaml` — confirm Alpha is listed, you're pending
7. Append yourself to `agents:` list with your discovered JSONL path
8. Read `turns/001_alpha_*.md` + Alpha's JSONL around `reply_marker`
9. Drop `turns/002_bravo_BOOTSTRAP.md` — acknowledge + first substantive reply per their `expects_next`
10. Your reply body includes HANDSHAKE_TOKEN

### Alpha, second turn (after Bravo joins)
1. Read session.yaml — confirm Bravo filled in
2. Read Bravo's turn marker + JSONL slice
3. Proceed per protocol from here

## 12. Protocol sanity: what to do if VERIFY fails

If you can't find the other agent's JSONL / turn marker / expected artifact:

1. Don't fabricate content. Your VERIFY step has failed.
2. Drop a turn marker with `sub_protocol: BLOCK`, `status: blocker`:
   ```yaml
   sub_protocol: BLOCK
   blocker_reason: "Bravo's JSONL at <path> not found. Confirm path or refresh."
   expects_next: "Human or Bravo to clarify path / restore state"
   ```
3. This escalates to human (you) automatically — you see BLOCK status in the latest turn marker.

## 13. Sub-agent vs collaborator distinction

- A **collaborator** is an agent listed in `session.yaml.agents` — they write turn markers, participate in the protocol.
- A **sub-agent** is spawned via `Task` tool internally by either collaborator. Sub-agents don't write turn markers; their output is consumed by the parent collaborator and summarized in the parent's turn marker if relevant.

Sub-agents are INTERNAL. If Alpha spawns a sub-agent to research something, Alpha's turn marker says "delegated research to sub-agent, findings summarized below" — it doesn't appear as a new `agent` in session.yaml.
