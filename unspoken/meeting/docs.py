"""Local document search: chunked .md/.txt files from config/meeting.toml docs_dir, cosine over the
registry's embed role when it is enabled, a keyword score otherwise. Nothing leaves the box."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CHUNK_CHARS = 700
_WORD = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "we", "our", "what", "whats",
        "how", "do", "does", "did", "it", "its", "that", "this", "about", "with", "be", "by", "at", "as", "from", "last",
        "time", "us", "you", "i", "can", "me", "tell", "again", "decide", "decided", "say", "said"}


@dataclass
class Chunk:
    doc: str
    index: int
    text: str
    vector: list[float] | None = None


@dataclass
class DocIndex:
    docs_dir: str
    embedder: Any = None                      # object with async embed(texts) -> vectors, or None
    chunks: list[Chunk] = field(default_factory=list)
    embedded: bool = False

    def load(self) -> "DocIndex":
        self.chunks = []
        folder = Path(self.docs_dir)
        if not folder.is_dir():
            return self
        for path in sorted(list(folder.glob("*.md")) + list(folder.glob("*.txt"))):
            text = path.read_text(encoding="utf-8", errors="replace")
            for i, piece in enumerate(chunk_text(text)):
                self.chunks.append(Chunk(path.name, i, piece))
        return self

    async def ensure_vectors(self) -> bool:
        if self.embedded or self.embedder is None or not self.chunks:
            return self.embedded
        try:
            vectors = await self.embedder.embed([c.text for c in self.chunks])
            for c, v in zip(self.chunks, vectors):
                c.vector = list(v)
            self.embedded = all(c.vector for c in self.chunks)
        except Exception:
            self.embedded = False
        return self.embedded

    async def search(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        """[{"doc", "chunk", "score", "text", "method"}], best first."""
        if not self.chunks:
            return []
        method = "keyword"
        scored: list[tuple[float, Chunk]] = []
        if await self.ensure_vectors():
            try:
                qv = (await self.embedder.embed([query]))[0]
                scored = [(cosine(qv, c.vector or []), c) for c in self.chunks]
                method = "embed"
            except Exception:
                scored = []
        if not scored:
            scored = [(keyword_score(query, c.text), c) for c in self.chunks]
        scored.sort(key=lambda sc: sc[0], reverse=True)
        out = []
        for score, c in scored[:k]:
            if score <= 0:
                continue
            out.append({"doc": c.doc, "chunk": c.index, "score": round(float(score), 4), "text": c.text, "method": method})
        return out


def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paragraphs:
        if cur and len(cur) + len(p) + 2 > size:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in STOP]


def keyword_score(query: str, text: str) -> float:
    q = tokens(query)
    if not q:
        return 0.0
    body = tokens(text)
    if not body:
        return 0.0
    counts: dict[str, int] = {}
    for w in body:
        counts[w] = counts.get(w, 0) + 1
    hits = 0.0
    for w in set(q):
        c = counts.get(w, 0)
        if c == 0 and len(w) > 4:                  # crude stemming: "pricing" ~ "price"
            c = sum(v for k, v in counts.items() if k.startswith(w[:5]))
        if c:
            hits += 1 + math.log(1 + c)
    return hits / (len(set(q)) + math.log(1 + len(body)) / 4)


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def answer_from(hits: list[dict[str, Any]], question: str, max_chars: int = 320) -> str:
    """The sentence(s) of the best chunk that share the most words with the question."""
    if not hits:
        return "I found nothing about that in the local docs."
    best = hits[0]
    body = "\n".join(line for line in best["text"].splitlines() if not line.lstrip().startswith("#"))
    sentences = [s.strip("- ").strip() for s in re.split(r"(?<=[.!?。])\s+|\n", body.replace("\n\n", "\n")) if s.strip("- ").strip()]
    q = set(tokens(question))
    sentences.sort(key=lambda s: len(q & set(tokens(s))), reverse=True)
    text = " ".join(sentences[:2]).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return f"{text} ({best['doc']})"
