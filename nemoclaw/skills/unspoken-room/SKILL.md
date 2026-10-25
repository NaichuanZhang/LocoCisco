---
name: unspoken-room
description: Answer questions about the live meeting (who is here, what was said, decisions, tasks, recap), check people in, pause or resume notes, undo items and propose tasks or lamp scenes through the room's `unspoken` MCP tools. Off-record content never exists.
user-invocable: true
---

# unspoken-room

Use this skill whenever a message concerns the meeting room: participants, the transcript, decisions, tasks,
notes being on or off the record, lights, the recap, or a reaction on a task or approval message.

## Procedure
1. `room_status` first. Read `privacy.state`: `on` (normal), `paused` or `off_record` (say so first).
2. Pick exactly one tool for the question (table in TOOLS.md). Do not chain tools unless the first one asks for
   an id you do not have (then `list_items`).
3. Reply in one or two sentences, or the recap markdown unchanged. Report tool verdicts word for word:
   `act` → "Done.", `ask_owner` → "Asked <owner> to approve.", `deny` → "The room declined.", `pending` → "Sent; no answer yet."

## Triggers
- Check-in: "I'm Priya", "Priya here", "/here Priya" → `checkin`.
- Privacy: "off the record", "/offrecord", "stop taking notes" → `pause_notes` immediately, then "Notes paused."
  "back on the record", "/resume" → `resume_notes`.
- Items: "undo", "scrap that", ❌ on a task → `undo_item`; ✅/❌ on an approval → `decide_request`.
- Questions: "what did we decide" → `list_items(kind="decision")`; "what did X say about Y" → `transcript(15)`;
  "where is the doc" → `search_docs`; "recap" → `recap`.
- Proposals: "add a task", "remind X to", "lights for slides", "dim the lights" → `create_task` / `set_light`.

## Never
- Never describe, guess or reconstruct anything said while notes were paused or off the record.
- Never paste the whole transcript; never invent owners, dates or quotes; never @everyone.
- Never claim an action happened unless the tool said `act` / `ok: true`.
