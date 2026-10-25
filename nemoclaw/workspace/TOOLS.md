# TOOLS.md — the room's tools (MCP server `unspoken`)

All tools run on the room's box. They never return off-record content because it was never stored.

| Tool | Use for | Notes |
|---|---|---|
| `room_status()` | title, topic, minutes left, who is checked in, privacy state, lamp/fan state, open item counts, egress meter | call first |
| `transcript(since_minutes=10)` | recent on-record lines `[{t, name, text}]` | max 120 minutes; quote sparingly |
| `list_items(kind="task"\|"decision"\|"question"\|"parking_lot"\|"all")` | captured items with `item_id`, owner, due, status | ids go in your posts: `(task t1)` |
| `search_docs(query, k=3)` | the room's local documents | answer from `hits[].text`, name `hits[].doc` |
| `recap()` | Discord-ready markdown recap from the local model | post unchanged |
| `checkin(name, user)` | "@LocoCisco I'm Priya" | binds the current unknown voice/seat for this meeting only |
| `pause_notes(reason, user)` | "off the record", /offrecord | always allowed; do it before replying |
| `resume_notes(user)` | back on the record | the policy may ask the owner |
| `create_task(text, owner, due, user)` | a task someone types | USER proposal; report the verdict |
| `set_light(scene, user)` | talk, slides, break, privacy, off_record, bright, dim | USER proposal; report the verdict |
| `undo_item(item_id, user)` | ❌ on a task, "undo that" | |
| `decide_request(request_id, approve, user)` | ✅ / ❌ on an approval the room asked for | |

`user` is always the Discord display name of the person who asked. Verdict words: `act` = done,
`ask_owner` = the meeting owner must approve, `deny` = the room declined, `pending` = no answer yet (say so).

Sending to Discord: the room posts its own notes, tasks and recaps through the bot (you will see them as your
own messages). Reactions on those messages reach you as events; handle ✅ ❌ per AGENTS.md.
