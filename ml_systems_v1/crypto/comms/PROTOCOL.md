# Agent-to-Agent Collaboration Protocol v2

Peer collaboration between Claude instances where **each agent reads the other's actual chat transcript from disk**. Human is not a content courier — human only signals "your turn, go."

## Core model

- **Session**: a named collaboration with a registered list of collaborators (agents) and their chat transcript file paths.
- **Turn**: one agent's turn = a verify → ideate → reply cycle. The agent's REPLY lives in their own chat (the normal chat output). They also drop a **turn marker** file as a pointer so the other side knows where to look.
- **Reading each other**: each agent knows the OTHER agent's chat transcript path (JSONL file under `.claude/projects/.../{session-uuid}.jsonl`) and reads it directly. No paste, no summarization by the human.
- **Human role**: spins up both chats, declares session + collaborators once, then just says "go" when it's a given agent's turn.

## Discovering the other agent's transcript

Claude transcripts live at:
```
~/.claude/projects/<slugified-cwd>/<session-uuid>.jsonl
```

On Windows:
```
C:\Users\<user>\.claude\projects\c--Users-<user>-Documents-coding-v4-crypto-stystem\<session-uuid>.jsonl
```

Each JSONL line is one turn (user prompt or assistant reply) with timestamps and content. An agent can locate the other agent's JSONL by:
1. Reading `comms/SESSIONS/{session_id}/session.yaml` → has `agents[*].jsonl_path`
2. Tail the specified JSONL file to find the most recent `"role": "assistant"` entry
3. That's the other agent's latest reply. Full content is in `content[*].text`

To find the OWN transcript (so it can record its path on join): look at the most recently modified `.jsonl` in the project's `.claude/projects/` folder (the one the current session is writing to), OR the harness may expose the path as an env var in some setups.

## The 3-step per-turn protocol

Every time an agent is activated with "your turn":

### Step 1: VERIFY
- Read `comms/SESSIONS/{id}/session.yaml` for collaborator list + roles
- Read the most recent turn marker(s) in `comms/SESSIONS/{id}/turns/` (newest first) to understand sequence + latest sub-protocol chosen
- Read the OTHER agent's JSONL — at minimum their most recent assistant reply; optionally prior exchanges for context
- **Side-channel rule (v2.1)**: read the OTHER agent's JSONL *user* messages since their last assistant reply to YOU. The human may have injected directives, clarifications, or constraint changes that shaped the other agent's reply but are NOT in their turn marker. The turn marker is a SUMMARY; the side-channel has the full provenance. You must process these human-injected messages as binding context for your own turn.
- Inspect any artifacts they touched (files listed in their turn marker)
- **Paranoid defaults**: where the other agent made numerical claims, verify by re-computing independently. If they said "Sharpe 3.14" from some script, run the check yourself on the persisted output.

### Step 2: IDEATE
- Decide what this turn IS:
  - Agree + add? (REVIEW-positive)
  - Disagree + correct? (REVIEW-negative → DEBATE)
  - Extend + build on? (PROPOSE on top)
  - Delegate next piece? (REQUEST)
  - Report completed work? (REPORT)
  - Ask for clarification? (QUESTION)
  - Concede + close? (CONCEDE)
  - Surface a blocker? (BLOCK)
  - Finalize? (DECISION)
- Pick the **sub-protocol** (one of the primitives above) that best fits. Dynamic — no single fixed structure.
- If you choose something adversarial (REVIEW-negative, DEBATE, BLOCK), have receipts: specific lines, specific re-computations, specific counterexamples.

### Step 3: REPLY
- Write the response in your chat (normal Claude output). **This is the canonical content.** Be thorough; the other side will read it from the JSONL.
- Drop a turn marker: `comms/SESSIONS/{id}/turns/NNN_{agent}_{sub_protocol}.md` — short pointer file with frontmatter + ≤200-word summary.
- **Record human-side-channel messages** in the turn marker under `human_directives_received:` (even if already applied). This flags to the other agent that they must read that slice of your JSONL during their VERIFY.
- Commit or save any artifacts. Reference them by path in the turn marker.
- **Human summary block**: if the session constitution has `human_summary_in_every_turn: true`, end your chat reply with a `## Human Summary` section (plain English, 1-3 short paragraphs, state + questions + next step). This is for the user, not the peer agent — but the peer should also skim it since it's the cleanest indicator of what the user understood.
- Your reply body should END with a short block saying "@<other_agent> — expects_next: <what you want from them>." so they have a clear ask when they activate.

## Turn marker format

Path: `comms/SESSIONS/{session_id}/turns/{NNN}_{from_agent}_{sub_protocol}.md`

```yaml
---
session_id: strategy-v2-research-001
turn: 7
from: Alpha
to: Bravo                                # or 'all' if multi-party
parent_turn: 6                           # which turn this responds to (null for session-opening)
sub_protocol: REVIEW                     # chosen dynamically per the IDEATE step
status: requires_response                # requires_response | informational | terminal
jsonl_path: "C:/Users/karab/.claude/projects/c--Users-karab-Documents-coding-v4-crypto-stystem/{uuid}.jsonl"
reply_marker: "2026-04-24T14:03:21Z"     # timestamp of most recent assistant reply in jsonl
artifacts_touched:
  - scripts/new_thing.py
  - docs/NEW_DOC.md
verifications_run:
  - "shuffle-entry control on asym_breakout, gap 7.78 Sharpe (confirmed signal real)"
  - "independent recompute of 4-sleeve blend — matches aggregator within rounding"
expects_next: Bravo to run low-first exit-order comparison + report
---

## Summary

One or two paragraphs telling the other agent what to expect before they load the full JSONL. Include:
- The sub_protocol you chose and why
- The key conclusion or ask
- Where the full detail lives (your JSONL around the reply_marker timestamp)

Don't duplicate the full reply here — the JSONL is the source of truth. This summary is a READER'S INDEX.
```

## Session file format

Path: `comms/SESSIONS/{session_id}/session.yaml`

```yaml
session_id: strategy-v2-research-001
started: 2026-04-24T13:50:00Z
status: active                           # active | paused | ended
purpose: >
  Validate and extend the asymmetric strategy frontier. Current scope:
  Family F (cross-asset dispersion), +120% target follow-through bounce
  redesign, and tier 2 live-paper validation.

agents:
  - name: Alpha
    role: Researcher                     # from ROLES.md templates
    specialization: [asymmetric strategy design, panel construction]
    jsonl_path: "C:/Users/karab/.claude/projects/c--.../{alpha-uuid}.jsonl"
    joined: 2026-04-24T13:50:00Z

  - name: Bravo
    role: Validator
    specialization: [paranoid validation, shuffle controls, exit-order audits]
    jsonl_path: "C:/Users/karab/.claude/projects/c--.../{bravo-uuid}.jsonl"
    joined: 2026-04-24T13:52:00Z

current_turn: 7                          # incremented on each turn marker write
next_up: Bravo                           # who's expected to go next
constitution:
  - paranoid_defaults_on_review: true    # Step 1 VERIFY must include independent compute
  - max_debate_rounds_before_human: 3
  - disagreement_protocol: REVIEW -> counter-REVIEW -> either CONCEDE or BLOCK-to-human
```

## Sub-protocols (your dynamic palette in Step 3)

Pick whichever fits the turn. Same names as v1 but now chosen per-turn rather than locked to a thread.

| Sub-protocol | When |
|--------------|------|
| **PROPOSE** | new idea / new design |
| **REVIEW** | evaluating other agent's work (positive or negative) |
| **REQUEST** | asking other agent to do a piece of work |
| **REPORT** | completed work, here's the outcome |
| **QUESTION** | need clarification before I can proceed |
| **ANSWER** | responding to a QUESTION |
| **DEBATE** | explicitly disagreeing, making a counter-case |
| **CONCEDE** | "you're right, I change my position" |
| **BLOCK** | "I can't proceed because X; needs human or major rethink" |
| **DECISION** | terminal — final call is logged, session may close |
| **HANDOFF** | transfer ownership of a workstream |

You can COMBINE them ("REVIEW + PROPOSE": critique their work, then propose an improvement). Note both in `sub_protocol`.

## Rules of engagement

1. **VERIFY is non-negotiable.** Step 1 always runs before you reply. No shortcut.
2. **Read the OTHER agent's latest JSONL reply before responding.** Don't reason from just the summary or the turn marker.
3. **One turn marker per turn.** If you realize mid-turn you need to revise, write a second marker with `supersedes: N-1` note.
4. **Artifacts commit with your turn.** Don't modify shared files without noting in `artifacts_touched`.
5. **Paranoid defaults when reviewing numbers.** Re-run enough of their claim to verify magnitude. Don't re-run purely symbolic work.
6. **Disagreements follow the constitution.** Usually ≤3 debate rounds before escalating to human.
7. **Session can host >2 agents.** `to:` field supports a list. `all` = broadcast.
8. **A dormant agent is not the same as an absent one.** If Alpha doesn't respond for >1h, Bravo can continue but notes "proceeded without Alpha's review" in their marker.

## Conflict handling

- **Same file, both edited**: the later writer wins the content, but flags the earlier's change by reference in their reply. If the earlier's work was materially broken, that's a BLOCK to them.
- **Numerical claim disagreement**: paranoid re-compute by both, cite sources, one usually concedes within 2 rounds.
- **Design disagreement with no objective winner**: DEBATE → if unresolved after 3 rounds, both file an ADVISORY to human with their recommendations. Human picks.

## Session lifecycle

1. **CREATE**: First agent drops `comms/SESSIONS/{id}/session.yaml` (or human does). Their agent info is filled in.
2. **JOIN**: Each subsequent agent appends themselves to `agents:` list.
3. **TURNS**: Agents alternate per `next_up`. Each writes a turn marker + updates `current_turn` and `next_up`.
4. **PAUSE/RESUME**: `status: paused` if human needs to inject context; resume with `status: active`.
5. **END**: One agent writes a DECISION-type turn marker. `status: ended`. The entire `SESSIONS/{id}/` is preserved as archive.

## Human's minimal role

You (human) provide:
- **Session creation prompt** once: "Set up session strategy-v2. You're Alpha; collaborator is Bravo at JSONL path X. Purpose: …"
- **Turn signals**: "You're up." In each agent's chat, at the start of their turn.
- **Context injection** when specifically needed (e.g. "external data update: new fetches landed")
- **Conflict resolution** if agents escalate (≤3 debate rounds, then you decide)

You do NOT need to:
- Paste content between chats
- Summarize for either agent
- Interpret messages

## Bootstrapping checklist

Human does once per session:

1. Open Chat 1. Tell it: "Your role is Alpha. Session `{id}` is created. You'll find me (Bravo) at `{jsonl_path}` — but since it doesn't exist yet, wait for my bootstrap. Protocol is in `comms/PROTOCOL.md`. Read it, then write an initial session.yaml + bootstrap turn-marker announcing yourself."
2. Open Chat 2. Tell it: "Your role is Bravo. Session `{id}` already exists at `comms/SESSIONS/{id}/`. Alpha's JSONL is at `{alpha_jsonl_path}`. Read PROTOCOL.md, read session.yaml, read Alpha's turn marker, read Alpha's JSONL reply, then drop your own bootstrap turn-marker appending yourself to agents: list."
3. Say "go" to Alpha. Alpha runs 3-step, replies, writes turn marker.
4. Say "go" to Bravo. Bravo runs 3-step, replies, writes turn marker.
5. Alternate. Intervene only when constitution specifies.

## Getting JSONL paths

On any Claude session, to discover its own JSONL path:
```bash
ls -t ~/.claude/projects/<slug>/*.jsonl | head -1
```
…or the harness may expose the path directly. If neither, the agent writes a distinctive string in its first message and the OTHER agent can `grep -l "distinctive string" ~/.claude/projects/<slug>/*.jsonl` to locate.

## See also

- [HANDSHAKE.md](HANDSHAKE.md) — operational specifics (JSONL discovery, reading slices, bootstrap ordering, race rules, cross-instance directive integration, VERIFY budget, WIP signal, artifact existence check)
- [ROLES.md](ROLES.md) — role templates for agents
- [EXAMPLE.md](EXAMPLE.md) — worked session example
- [SESSIONS/](SESSIONS/) — active and archived sessions

## Relationship to existing cross-instance directive

`memory/MEMORY.md` has a directive: *"session start + every ~30min, delta-check `~/.claude/projects/<slug>/*.jsonl` mtimes + `git log --since='1 hour ago'` + `git status`."*

The comms protocol REFINES this for registered collaborators. Rules:
- Collaborators in `session.yaml` are tracked by turn markers (not mtime scan)
- THIRD-PARTY jsonls discovered via mtime scan (instances not in session) are noted in `external_context_seen:` field of your turn marker
- Third-party work may be referenced but doesn't bind the session
- If external work materially affects the session, propose human adds them as a collaborator OR note their work as context
