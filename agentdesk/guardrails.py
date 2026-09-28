"""Safety checks that run around the agents.

- detect_injection: flags text that tries to override the assistant's instructions.
  Applied to user messages and to every retrieved document chunk.
- redact_pii: masks emails, phone numbers and card numbers before text is logged
  or returned.
- The tool permission and approval rules live in tools.py, next to the tools.
"""

from __future__ import annotations

import re

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|rules|messages)",
    r"disregard (all |any )?(the )?(previous|prior|above|your) (instructions|rules)",
    r"forget (all |your )?(previous |prior )?(instructions|rules)",
    r"you are now (a|an|in) ",
    r"(reveal|show|print|repeat) (me )?(your|the) (system prompt|instructions|hidden prompt)",
    r"\bsystem prompt\b",
    r"developer mode",
    r"\bjailbreak\b",
    r"act as (an? )?(admin|administrator|supervisor|developer)",
    r"(approve|issue) (a )?(full )?refunds? for (every|all) orders?",
    r"without (asking|approval from) (a )?supervisor",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+91[ -]?)?[6-9]\d{9}(?!\d)")


def detect_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text or ""))


def redact_pii(text: str) -> str:
    text = _EMAIL_RE.sub(lambda m: _mask_email(m.group()), text)
    text = _CARD_RE.sub(lambda m: _mask_card(m.group()), text)
    text = _PHONE_RE.sub(lambda m: "******" + re.sub(r"\D", "", m.group())[-4:], text)
    return text


def _mask_email(email: str) -> str:
    user, _, domain = email.partition("@")
    return f"{user[0]}***@{domain}"


def _mask_card(number: str) -> str:
    digits = re.sub(r"\D", "", number)
    # Order numbers and other short ids are left alone; only card-length runs are masked.
    if len(digits) < 13:
        return number
    return "**** **** **** " + digits[-4:]
