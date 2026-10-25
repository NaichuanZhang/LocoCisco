"""Method `llm`: DiarizationLM-style correction of speaker boundaries from the text alone.

DiarizationLM (Wang et al. 2024, https://arxiv.org/abs/2401.03506) shows a text LLM can repair word-level
speaker errors around turn boundaries ("...so maybe we start with you. I'm" | "the industrial designer" ->
the "I'm" belongs to the second speaker). This is the local, zero-shot version: for every boundary between two
utterances of different speakers, the LLM sees the last `llm_window_words` words of the first speaker and the
first `llm_window_words` of the second, tagged, and answers with the same words re-tagged. Only the boundary
position moves (words never change, the two speakers never change), by at most the window; a reply that is not
a permutation of the shown words is ignored. Words move with their timings when the ASR has word timings;
otherwise the utterance times are split in proportion to the characters moved. `Transcript.meta["llm"]` reports
calls, moved words and latency.

Client: llama-server (llama.cpp) on loopback, OpenAI-compatible /v1/chat/completions, thinking disabled
(Qwen3 `/no_think` + chat_template_kwargs). Any object with `.complete(prompt) -> str` works (tests use a fake).
"""
from __future__ import annotations

import json
import re
import statistics
import time
from typing import Any

import httpx

from .contracts import Transcript, Utterance, Word
from .fuse import Settings

SYSTEM = ("You fix speaker boundaries in a meeting transcript. You get two consecutive speaker turns as tags. "
          "Some words right at the boundary may be tagged to the wrong speaker. Output the SAME words in the SAME "
          "order, with the tags moved so each word is with the speaker who most plausibly said it. Never add, drop "
          "or change words. Never swap the two speakers. If the boundary is already right, output it unchanged. "
          "Output only the tagged text, nothing else. /no_think")


class LlamaClient:
    def __init__(self, base_url: str, timeout: float = 180.0, max_tokens: int = 160, model: str = "local") -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
        self.max_tokens = max_tokens
        self.model = model

    def complete(self, prompt: str) -> str:
        body = {"model": self.model, "temperature": 0.0, "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                "chat_template_kwargs": {"enable_thinking": False}}
        r = self.client.post(f"{self.base_url}/v1/chat/completions", json=body)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    def healthy(self) -> bool:
        try:
            return self.client.get(f"{self.base_url}/health", timeout=3).status_code == 200
        except httpx.HTTPError:
            return False


_TAG = re.compile(r"<(spk[12])>(.*?)</\1>", re.S)
_WORD = re.compile(r"\S+")


def tagged(a: list[str], b: list[str]) -> str:
    return f"<spk1> {' '.join(a)} </spk1> <spk2> {' '.join(b)} </spk2>"


def parse_reply(reply: str, shown: list[str]) -> int | None:
    """Number of leading words tagged spk1 in the reply, when the reply holds exactly the shown words in order."""
    tags = _TAG.findall(reply)
    if not tags:
        return None
    words, n1 = [], 0
    for tag, body in tags:
        ws = _WORD.findall(body)
        if tag == "spk1":
            if words and words[-1][0] == "spk2":
                return None                                # spk1 after spk2 = speakers swapped or interleaved
            n1 += len(ws)
        words += [(tag, w) for w in ws]
    if [w for _, w in words] != shown:
        return None
    return n1


def _move(a: Utterance, b: Utterance, k: int) -> None:
    """k > 0: the first k words of b go to the end of a; k < 0: the last -k words of a go to the front of b."""
    aw, bw = a.text.split(), b.text.split()
    if k > 0:
        moved, bw = bw[:k], bw[k:]
        aw += moved
    else:
        moved, aw = aw[k:], aw[:k]
        bw = moved + bw
    if not aw or not bw:
        return
    if a.words and b.words:
        if k > 0:
            mv, b.words = b.words[:k], b.words[k:]
            a.words += [Word(w.start, w.end, w.text, w.conf, a.speaker) for w in mv]
        else:
            mv, a.words = a.words[k:], a.words[:k]
            b.words = [Word(w.start, w.end, w.text, w.conf, b.speaker) for w in mv] + b.words
        a.end, b.start = round(max(w.end for w in a.words), 3), round(min(w.start for w in b.words), 3)
    else:
        src = b if k > 0 else a
        chars_total = max(len(src.text), 1)
        chars_moved = len(" ".join(moved))
        dt = (src.end - src.start) * chars_moved / chars_total
        if k > 0:
            cut = round(b.start + dt, 3)
            a.end, b.start = max(a.end, cut), cut
        else:
            cut = round(a.end - dt, 3)
            a.end, b.start = cut, min(b.start, cut)
    a.text, b.text = " ".join(aw), " ".join(bw)


def correct_boundaries(tr: Transcript, llm: Any, s: Settings) -> Transcript:
    t0 = time.perf_counter()
    report: dict[str, Any] = {"boundaries": 0, "asked": 0, "moved": 0, "words_moved": 0, "unparsed": 0,
                              "skipped_no_client": llm is None}
    if llm is None:
        tr.meta["llm"] = report
        return tr
    utts = sorted(tr.utterances, key=lambda u: u.start)
    tr.utterances = utts
    lat: list[float] = []
    changes: list[dict[str, Any]] = []
    for i in range(len(utts) - 1):
        a, b = utts[i], utts[i + 1]
        if a.speaker == b.speaker:
            continue
        report["boundaries"] += 1
        if report["asked"] >= s.llm_max_calls:
            continue
        aw, bw = a.text.split(), b.text.split()
        if len(aw) < 2 and len(bw) < 2:
            continue
        wa, wb = aw[-s.llm_window_words:], bw[:s.llm_window_words]
        prompt = tagged(wa, wb)
        t1 = time.perf_counter()
        try:
            reply = llm.complete(prompt)
        except Exception as exc:
            report.setdefault("errors", []).append(f"{i}: {type(exc).__name__}: {exc}")
            continue
        lat.append((time.perf_counter() - t1) * 1000)
        report["asked"] += 1
        n1 = parse_reply(reply, wa + wb)
        if n1 is None:
            report["unparsed"] += 1
            continue
        k = n1 - len(wa)                     # >0: words of b move to a; <0: words of a move to b
        if k == 0:
            continue
        if n1 == 0 or n1 == len(wa) + len(wb):
            continue                         # the model emptied a turn: not a boundary move, ignore
        _move(a, b, k)
        a.source = b.source = "llm"
        report["moved"] += 1
        report["words_moved"] += abs(k)
        changes.append({"i": i, "k": k, "prompt": prompt, "reply": reply})
    if lat:
        report["latency_ms"] = {"mean": round(statistics.mean(lat), 1), "median": round(statistics.median(lat), 1),
                                "max": round(max(lat), 1)}
    report["wall_s"] = round(time.perf_counter() - t0, 2)
    report["changes"] = changes[:50]
    tr.meta["llm"] = report
    tr.meta.setdefault("stages", []).append("llm")
    return tr
