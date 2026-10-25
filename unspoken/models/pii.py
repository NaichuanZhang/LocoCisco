"""Provider "regex" for the pii role: cheap, local, no model. Finds the spans that must never leave the box
in a Discord note: e-mail addresses, phone numbers, keys/tokens and money amounts.

It is deliberately conservative about phones (needs 9-15 digits with separators, so years, ticket ids and
room numbers are not flagged) and generous about secrets (any long key-looking token is one).
"""
from __future__ import annotations

import re

from ..contracts import PIISpan, RoleSpec

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# +1 415 555 0133 / (415) 555-0133 / 415-555-0133 / +44 20 7946 0958 / 010-1234-5678
PHONE = re.compile(r"(?<![\w.])(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?|\d{2,4}[\s.-])\d{3,4}[\s.-]?\d{3,4}(?![\w.])")
SECRET = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{16,}"                 # OpenAI-style
    r"|AKIA[0-9A-Z]{16}"                        # AWS access key id
    r"|gh[pousr]_[A-Za-z0-9]{30,}"              # GitHub tokens
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"            # Slack tokens
    r"|AIza[0-9A-Za-z_-]{35}"                   # Google API key
    r"|(?<![A-Za-z0-9/+=_-])[A-Fa-f0-9]{32,}(?![A-Za-z0-9/+=_-])"   # long hex (api keys, hashes)
    r"|Bearer\s+[A-Za-z0-9._~+/=-]{8,}"         # HTTP bearer token
    r"|(?:token|api[_-]?key|secret|password|passwd|pwd)\s*[:=]\s*\S{6,})",
    re.IGNORECASE)
MONEY = re.compile(
    r"(?:[$€£¥]\s?\d[\d,]*(?:\.\d+)?\s?(?:[kKmMbB](?![A-Za-z])|(?:thousand|million|billion)\b)?"
    r"|\b\d[\d,]*(?:\.\d+)?\s?(?:[kKmM]\s)?(?:USD|EUR|GBP|CNY|RMB|JPY|dollars?|euros?|pounds?|bucks|yuan)\b)",
    re.IGNORECASE)

PATTERNS: tuple[tuple[str, re.Pattern[str], float], ...] = (
    ("secret", SECRET, 0.95),
    ("email", EMAIL, 0.99),
    ("money", MONEY, 0.9),
    ("phone", PHONE, 0.8),
)


class RegexPII:
    def __init__(self, spec: RoleSpec | None = None) -> None:
        self.patterns = PATTERNS

    async def find(self, text: str) -> list[PIISpan]:
        spans: list[PIISpan] = []
        for label, pattern, score in self.patterns:
            for m in pattern.finditer(text):
                s, e = m.start(), m.end()
                if e - s < 3 or any(not (e <= o.start or s >= o.end) for o in spans):   # first match wins
                    continue
                spans.append(PIISpan(start=s, end=e, label=label, score=score))
        return sorted(spans, key=lambda sp: sp.start)

    async def healthy(self) -> bool:
        return True
