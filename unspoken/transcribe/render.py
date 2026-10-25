"""Render a Transcript (contracts.py) as readable text, SRT subtitles or markdown.

    python -m unspoken.transcribe.render transcript.json [--format txt|srt|md] [--out FILE]

txt   "[00:12.3] Left person: ..." one utterance per line (what the meeting screen shows)
srt   numbered cues "00:00:12,300 --> 00:00:15,900" with "Left person: ..." (any video player)
md    a table (time, who, text) with a speaker legend (seat, bound voice, on camera)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .contracts import Transcript, Utterance, load_transcript

FORMATS = ("txt", "srt", "md")


def _label(tr: Transcript, u: Utterance) -> str:
    info = tr.speakers.get(u.speaker)
    return info.label if info and info.label else u.speaker


def fmt_time(t: float) -> str:
    """[mm:ss.d] style: 72.34 -> '01:12.3'."""
    t = max(t, 0.0)
    m, sec = divmod(t, 60.0)
    return f"{int(m):02d}:{sec:04.1f}"


def fmt_srt(t: float) -> str:
    t = max(t, 0.0)
    h, rem = divmod(t, 3600.0)
    m, sec = divmod(rem, 60.0)
    return f"{int(h):02d}:{int(m):02d}:{int(sec):02d},{int(round((sec - int(sec)) * 1000)) % 1000:03d}"


def render_txt(tr: Transcript, show_conf: bool = False) -> str:
    lines = []
    for u in sorted(tr.utterances, key=lambda u: u.start):
        tail = f"  ({u.conf:.2f} {u.source})" if show_conf else ""
        lines.append(f"[{fmt_time(u.start)}] {_label(tr, u)}: {u.text.strip()}{tail}")
    return "\n".join(lines) + ("\n" if lines else "")


def render_srt(tr: Transcript) -> str:
    out = []
    for i, u in enumerate(sorted(tr.utterances, key=lambda u: u.start), 1):
        end = max(u.end, u.start + 0.5)
        out.append(f"{i}\n{fmt_srt(u.start)} --> {fmt_srt(end)}\n{_label(tr, u)}: {u.text.strip()}\n")
    return "\n".join(out)


def render_md(tr: Transcript) -> str:
    out = [f"# Transcript {tr.session_id}".rstrip(), "",
           f"Duration {fmt_time(tr.duration)}, {len(tr.utterances)} utterances, method `{tr.meta.get('method', '?')}`.", "",
           "| speaker | label | seat | on camera | voice |", "|---|---|---|---|---|"]
    for key, info in tr.speakers.items():
        out.append(f"| `{key}` | {info.label} | {info.seat or ''} | {'yes' if info.on_camera else 'no'} | {info.voice or ''} |")
    out += ["", "| time | who | said | conf |", "|---|---|---|---|"]
    for u in sorted(tr.utterances, key=lambda u: u.start):
        text = u.text.strip().replace("|", "\\|")
        out.append(f"| {fmt_time(u.start)} | {_label(tr, u)} | {text} | {u.conf:.2f} {u.source} |")
    return "\n".join(out) + "\n"


def render(tr: Transcript, fmt: str) -> str:
    if fmt == "txt":
        return render_txt(tr)
    if fmt == "srt":
        return render_srt(tr)
    if fmt == "md":
        return render_md(tr)
    raise ValueError(f"format must be one of {FORMATS}")


def render_files(tr: Transcript, stem: str | Path) -> list[Path]:
    """Writes <stem>.txt, <stem>.srt and <stem>.md; returns the paths."""
    stem = Path(stem)
    paths = []
    for fmt in FORMATS:
        p = stem.with_suffix(f".{fmt}")
        p.write_text(render(tr, fmt))
        paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transcript", type=Path)
    ap.add_argument("--format", default="txt", choices=FORMATS)
    ap.add_argument("--out", type=Path, help="write here instead of stdout")
    ap.add_argument("--conf", action="store_true", help="txt: append the speaker confidence and source")
    a = ap.parse_args(argv)
    tr = load_transcript(a.transcript)
    text = render_txt(tr, show_conf=True) if (a.format == "txt" and a.conf) else render(tr, a.format)
    if a.out:
        a.out.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
