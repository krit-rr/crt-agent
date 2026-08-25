"""Rule-based prose -> `ProblemSpec` mapping.

This is the pre-LLM way to do it, and it is here to be beaten. Each family gets a
regex; anything the regexes don't recognise produces `None`, and the agent that uses
this abstains rather than guessing.

Keeping it in the repo is not nostalgia. It is the floor of the comparison: it shows
exactly what the LLM parser buys you (robustness to phrasing) and exactly what it
costs you (nondeterminism, latency, the possibility of a wrong-but-well-formed spec).

The mock LLM provider also uses these rules, so the whole benchmark runs offline.
"""

from __future__ import annotations

import re
from typing import Any

from crt_agent.items.templates import TEMPLATES
from crt_agent.schemas import ItemFamily, ProblemSpec

_NUM = r"([\d,]+(?:\.\d+)?)"


def _f(text: str) -> float:
    return float(text.replace(",", ""))


PATTERNS: dict[ItemFamily, re.Pattern[str]] = {
    "bat_ball": re.compile(
        rf"A (?P<big>[\w ]+?) and a (?P<small>[\w ]+?) cost \${_NUM} in total\. "
        rf"The (?P=big) costs \${_NUM} more than the (?P=small)",
        re.I,
    ),
    "widgets": re.compile(
        rf"If {_NUM} (?P<worker>[\w ]+?)s take {_NUM} minutes to make {_NUM} (?P<thing>[\w ]+?)s, "
        rf"how many minutes would it take {_NUM} (?P=worker)s to make {_NUM} (?P=thing)s",
        re.I,
    ),
    "lily_pad": re.compile(
        r"Every day, the .+? (?P<growth>doubles|triples|quadruples|grows \d+-fold) in size\. "
        r"If it takes (?P<days>\d+) days",
        re.I,
    ),
    "discount": re.compile(
        rf"priced at \${_NUM}\. In a sale its price is cut by (?P<pct>\d+)%",
        re.I,
    ),
    "rank": re.compile(
        r"the (?P<n>\d+)(?:st|nd|rd|th) highest score and the (?P<m>\d+)(?:st|nd|rd|th) lowest",
        re.I,
    ),
    "drift": re.compile(
        r"has (?P<backlog>\d+) people on its waitlist\. It can see (?P<served>\d+) new people "
        r"per day, but (?P<arriving>\d+) new people join",
        re.I,
    ),
}

_GROWTH = {"doubles": 2, "triples": 3, "quadruples": 4}


def _params(family: ItemFamily, m: re.Match[str]) -> dict[str, Any]:
    if family == "bat_ball":
        return {
            "big": m.group("big"),
            "small": m.group("small"),
            "total": _f(m.group(3)),
            "diff": _f(m.group(4)),
        }
    if family == "widgets":
        return {
            "worker": m.group("worker"),
            "thing": m.group("thing"),
            "m": int(_f(m.group(1))),
            "t": int(_f(m.group(3))),
            "w": int(_f(m.group(4))),
            "M": int(_f(m.group(6))),
            "W": int(_f(m.group(7))),
        }
    if family == "lily_pad":
        growth = m.group("growth").lower()
        k = _GROWTH.get(growth) or int(re.search(r"\d+", growth).group())  # type: ignore[union-attr]
        return {"k": k, "D": int(m.group("days"))}
    if family == "discount":
        return {"price": _f(m.group(1)), "pct": int(m.group("pct"))}
    if family == "rank":
        return {"n": int(m.group("n")), "m": int(m.group("m"))}
    if family == "drift":
        return {
            "backlog": int(m.group("backlog")),
            "served": int(m.group("served")),
            "arriving": int(m.group("arriving")),
        }
    raise AssertionError(family)  # pragma: no cover


def match_spec(text: str) -> tuple[ItemFamily, ProblemSpec] | None:
    """Return the matched family and its spec, or `None` if no rule fires."""
    for family, pattern in PATTERNS.items():
        m = pattern.search(text)
        if not m:
            continue
        params = dict(TEMPLATES[family].canonical)
        params.update(_params(family, m))
        return family, TEMPLATES[family].spec(params)
    return None
