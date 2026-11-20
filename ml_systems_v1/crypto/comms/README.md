# comms/ — Agent-to-Agent Collaboration (v2)

Peer collaboration between Claude instances. **Agents read each other's chat transcripts directly from disk** — no copy-paste. Human just signals turn completion.

## How this differs from v1

| v1 (retired) | v2 (this) |
|--------------|-----------|
| Human copies message content between chats | Agent reads the OTHER agent's JSONL chat transcript directly |
| Messages in inbox/ as full-content markdown | Turn markers as pointers + summary; full reply lives in each agent's own chat |
| Threading via `thread:` field | Session-scoped under `SESSIONS/{id}/` |
| Fixed per-thread roles | Roles per session, sub-protocol dynamic per-turn |
| Fixed per-turn step | 3-step protocol: VERIFY → IDEATE → REPLY |

## Quick start (human)

1. **Open two Claude sessions** on this repo (Chat 1 + Chat 2).
2. **Find each chat's JSONL path** — on Windows: `C:/Users/<you>/.claude/projects/<slug>/<session-uuid>.jsonl`. Most-recently-modified .jsonl is the active session.
3. **Bootstrap Chat 1 (Alpha)**:
   > *"You are Alpha. Session ID: `<name>`. Role: <from ROLES.md>. Your JSONL path: `<chat1-jsonl>`. Read `comms/PROTOCOL.md` then create `comms/SESSIONS/<name>/session.yaml` with yourself, and drop a bootstrap turn-marker. Purpose: <what we're working on>."*
4. **Bootstrap Chat 2 (Bravo)**:
   > *"You are Bravo. Session `<name>` is at `comms/SESSIONS/<name>/`. Alpha's JSONL: `<chat1-jsonl>`. Your JSONL: `<chat2-jsonl>`. Read PROTOCOL.md, read the session, read Alpha's turn marker + their JSONL reply, append yourself to session.yaml agents list, drop your bootstrap turn-marker."*
5. **Then just say "your turn, go"** to whoever's up (per `next_up` in session.yaml). Repeat.

You never paste content. You only signal turn-completion.

## What each agent does per turn (the 3 steps)

1. **VERIFY** — read protocol, session state, other agent's turn marker, other agent's JSONL reply, any artifacts. Re-run numerical claims independently (paranoid defaults on).
2. **IDEATE** — decide what the turn is (PROPOSE / REVIEW / REQUEST / REPORT / QUESTION / ANSWER / DEBATE / CONCEDE / BLOCK / DECISION / HANDOFF). Dynamic choice.
3. **REPLY** — respond in your chat (canonical content lives in YOUR JSONL). Drop a turn marker at `comms/SESSIONS/<id>/turns/NNN_<name>_<subprotocol>.md`. Update `current_turn` and `next_up` in session.yaml.

## Files

- **PROTOCOL.md** — full spec (3-step turn, JSONL reading, session lifecycle, conflict rules)
- **ROLES.md** — role templates (peer / Maker-Breaker / Researcher-Validator / ...)
- **EXAMPLE.md** — worked 5-turn session showing the full loop
- **SESSIONS/** — active and archived sessions
  - `<session_id>/session.yaml` — collaborators + purpose + constitution
  - `<session_id>/turns/NNN_<name>_<subprotocol>.md` — turn markers (pointers + summaries)
  - `<session_id>/` may also contain session-scoped artifacts

## When to use

- Non-trivial work where a second perspective adds value
- Paranoid validation of numerical claims (Maker/Validator split)
- Parallel workstreams with a clean interface (domain split)
- High-uncertainty decisions (Debate pattern)

## When not to use

- Trivial tasks (one agent suffices)
- Tasks requiring constant back-and-forth without natural turn boundaries
- Tasks where one agent has strictly more context and the second would just echo

## Constitution defaults

Per-session in `session.yaml.constitution`:
```yaml
- paranoid_defaults_on_review: true     # Step 1 VERIFY must re-run numerical claims
- max_debate_rounds_before_human: 3     # after 3 unresolved counter-REVIEWs, escalate
- disagreement_protocol: "REVIEW -> counter-REVIEW -> CONCEDE or BLOCK-to-human"
```

Override per-session if needed.

## See also

- [PROTOCOL.md](PROTOCOL.md) — full spec
- [ROLES.md](ROLES.md) — role templates
- [EXAMPLE.md](EXAMPLE.md) — worked example
