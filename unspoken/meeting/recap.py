"""Recap: decisions, tasks with owners and due dates, open questions, as Discord-ready markdown.

The template is always available and is what the demo shows when the llm role is disabled; with the
local LLM enabled, a short Strands call polishes the template plus the on-record transcript into prose
(and falls back to the template on any error). Off-record and wiped lines were deleted before this runs,
so they cannot appear here.
"""
from __future__ import annotations

from typing import Any

from .base import MeetingState
from .dates import humanize

MAX_TRANSCRIPT_CHARS = 6000


def recap_template(state: MeetingState) -> str:
    title = state.status.get("title") or "Meeting"
    owner = state.status.get("owner")
    lines = [f"**Recap: {title}**" + (f" (owner: {owner})" if owner else "")]
    names = state.names()
    if names:
        lines.append("Here: " + ", ".join(names) + (f" + {state.guests_present()} guest(s)" if state.guests_present() else ""))
    decisions = [i for i in state.items.values() if i.get("kind") == "decision" and i.get("status") == "open"]
    tasks = [i for i in state.items.values() if i.get("kind") == "task" and i.get("status") in ("open", "pending_approval")]
    questions = [i for i in state.items.values() if i.get("kind") == "question" and i.get("status") == "open"]
    parked = [i for i in state.items.values() if i.get("kind") == "parking_lot" and i.get("status") == "open"]
    lines.append("")
    lines.append("**Decisions**")
    lines += [f"- {d['text']}" for d in decisions] or ["- none recorded"]
    lines.append("")
    lines.append("**Tasks**")
    lines += [f"- {t['text']} — {t.get('owner') or 'no owner yet'}, {humanize(t.get('due'))}" for t in tasks] or ["- none"]
    if questions:
        lines.append("")
        lines.append("**Open questions**")
        lines += [f"- {q['text']}" for q in questions]
    if parked:
        lines.append("")
        lines.append("**Parking lot**")
        lines += [f"- {p['text']}" for p in parked]
    no_owner = sum(1 for t in tasks if not t.get("owner"))
    if no_owner:
        lines.append("")
        lines.append(f"_{no_owner} task{'s' if no_owner > 1 else ''} still need{'s' if no_owner == 1 else ''} an owner._")
    return "\n".join(lines)


async def recap_markdown(state: MeetingState, model_factory: Any = None) -> str:
    """Template, optionally rewritten by the local LLM (model_factory() -> strands Model)."""
    template = recap_template(state)
    if model_factory is None:
        return template
    try:
        from strands import Agent
        model = model_factory()
        transcript = "\n".join(f"{(u.get('speaker') or {}).get('name') or 'Someone'}: {u.get('text', '')}"
                               for u in state.transcript)[-MAX_TRANSCRIPT_CHARS:]
        agent = Agent(model=model, tools=[], callback_handler=None, system_prompt=(
            "You write a crisp meeting recap in Discord markdown. Keep EVERY decision and task from the draft "
            "exactly (owners and due dates included), add one line of context per decision from the transcript "
            "when useful, and never add items that are not in the draft or the transcript. Under 25 lines."))
        result = await agent.invoke_async(f"Draft:\n{template}\n\nOn-record transcript:\n{transcript or '(none)'}")
        text = str(result).strip()
        return text or template
    except Exception:
        return template
