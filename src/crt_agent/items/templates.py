"""Parametric CRT item families.

Every family knows four things:

1. how to render itself as prose for a given set of parameters,
2. the correct answer,
3. the **lure** — the specific wrong answer System 1 produces,
4. a reference `ProblemSpec` (the formalisation we'd consider ideal).

Point 3 is the reason this file exists. A benchmark that only records right/wrong
cannot distinguish "the model is bad at arithmetic" from "the model fell into the
cognitive trap the item was designed to set". Point 4 lets us score a parser's
*formalisation* independently of whether the arithmetic came out right.

Because the parameters are free, the classic items ($1.10 bat, 48 days, 100 machines)
are just one point in a large space. Everything else is text that has never appeared
in any training corpus, while preserving the trap structure exactly.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from crt_agent.schemas import ItemFamily, ProblemSpec, Variable


def _money(x: float) -> str:
    return f"{x:,.2f}"


def _growth_phrase(k: int) -> str:
    return {2: "doubles", 3: "triples", 4: "quadruples"}.get(k, f"grows {k}-fold")


@dataclass(frozen=True)
class Template:
    family: ItemFamily
    canonical: dict[str, Any]
    render: Callable[[dict[str, Any]], str]
    answer: Callable[[dict[str, Any]], float]
    lure: Callable[[dict[str, Any]], float]
    spec: Callable[[dict[str, Any]], ProblemSpec]
    sample: Callable[[random.Random], dict[str, Any]]
    unit: str = ""
    trap: str = ""
    #: parameter keys that are *surface* (nouns, names) rather than numeric.
    #: Perturbing these alone changes the wording while leaving every number and the
    #: correct answer identical to the canonical item — which separates "recalled the
    #: wording" from "recalled the number".
    surface: tuple[str, ...] = ()


# --------------------------------------------------------------------------------------
# 1. Bat and ball  (Frederick 2005)
# --------------------------------------------------------------------------------------
# total = a + b, difference = a - b.  Answer (b) = (total - diff) / 2.
# Lure: subtract the difference from the total and stop.  b_lure = total - diff.

BAT_BALL = Template(
    family="bat_ball",
    unit="dollars",
    trap="Subtracting the difference from the total feels like the whole computation.",
    canonical={"total": 1.10, "diff": 1.00, "big": "bat", "small": "ball"},
    render=lambda p: (
        f"A {p['big']} and a {p['small']} cost ${_money(p['total'])} in total. "
        f"The {p['big']} costs ${_money(p['diff'])} more than the {p['small']}. "
        f"How much does the {p['small']} cost, in dollars?"
    ),
    answer=lambda p: (p["total"] - p["diff"]) / 2,
    lure=lambda p: p["total"] - p["diff"],
    spec=lambda p: ProblemSpec(
        variables=[
            Variable(name="big", description=f"cost of the {p['big']}", unit="dollars"),
            Variable(name="small", description=f"cost of the {p['small']}", unit="dollars"),
        ],
        equations=[f"big + small = {p['total']}", f"big - small = {p['diff']}"],
        query="small",
        unit="dollars",
        assumptions=["'more than' is an additive difference, not a ratio"],
    ),
    sample=lambda rng: _sample_bat_ball(rng),
    surface=("big", "small"),
)


def _sample_bat_ball(rng: random.Random) -> dict[str, Any]:
    big, small = rng.choice(
        [
            ("laptop", "sleeve"),
            ("headset", "cable"),
            ("bike", "helmet"),
            ("desk", "lamp"),
            ("racket", "shuttlecock"),
            ("kettle", "mug"),
        ]
    )
    diff = rng.randrange(50, 900) / 100  # 0.50 .. 8.99
    gap = rng.randrange(2, 60) / 100  # what the small item costs, doubled
    total = round(diff + 2 * gap, 2)
    return {"total": total, "diff": diff, "big": big, "small": small}


# --------------------------------------------------------------------------------------
# 2. Machines and widgets
# --------------------------------------------------------------------------------------
# m machines make w widgets in t minutes  ->  per-machine rate = w / (m * t).
# Answer for M machines / W widgets: time = W * m * t / (M * w).
# Lure: assume duration scales with the widget count -> t * W / w.

WIDGETS = Template(
    family="widgets",
    unit="minutes",
    trap="The numbers invite you to scale time with the count instead of holding the rate fixed.",
    canonical={"m": 5, "w": 5, "t": 5, "M": 100, "W": 100, "thing": "widget", "worker": "machine"},
    render=lambda p: (
        f"If {p['m']} {p['worker']}s take {p['t']} minutes to make {p['w']} {p['thing']}s, "
        f"how many minutes would it take {p['M']} {p['worker']}s to make {p['W']} {p['thing']}s?"
    ),
    answer=lambda p: p["W"] * p["m"] * p["t"] / (p["M"] * p["w"]),
    lure=lambda p: p["t"] * p["W"] / p["w"],
    spec=lambda p: ProblemSpec(
        variables=[
            Variable(
                name="rate",
                description=f"{p['thing']}s produced per {p['worker']} per minute",
                unit="per minute",
            ),
            Variable(name="minutes", description="time for the second scenario", unit="minutes"),
        ],
        equations=[
            f"rate * {p['m']} * {p['t']} = {p['w']}",
            f"rate * {p['M']} * minutes = {p['W']}",
        ],
        query="minutes",
        unit="minutes",
        assumptions=[f"every {p['worker']} works at the same constant rate, in parallel"],
    ),
    sample=lambda rng: _sample_widgets(rng),
    surface=("worker", "thing"),
)


def _sample_widgets(rng: random.Random) -> dict[str, Any]:
    worker, thing = rng.choice(
        [
            ("machine", "widget"),
            ("printer", "poster"),
            ("baker", "bun"),
            ("nurse", "screening"),
            ("scanner", "page"),
            ("robot", "panel"),
        ]
    )
    n = rng.randrange(3, 13)
    t = rng.randrange(2, 12)
    k = rng.choice([4, 5, 8, 10, 12, 20, 25])
    return {"m": n, "w": n, "t": t, "M": n * k, "W": n * k, "thing": thing, "worker": worker}


# --------------------------------------------------------------------------------------
# 3. Lily pad  (exponential growth)
# --------------------------------------------------------------------------------------
# Area multiplies by k each day and fills the lake on day D.
# Answer for the day it covered 1/k of the lake: D - 1.  Lure: D / 2.

LILY_PAD = Template(
    family="lily_pad",
    unit="days",
    trap="Halving the *time* feels like the natural inverse of halving the *area*.",
    canonical={"k": 2, "D": 48, "frac": "half", "thing": "patch of lily pads", "place": "lake"},
    render=lambda p: (
        f"In a {p['place']} there is a {p['thing']}. Every day, the {p['thing']} "
        f"{_growth_phrase(p['k'])} in size. "
        f"If it takes {p['D']} days for the {p['thing']} to cover the entire {p['place']}, "
        f"how many days would it take to cover {p['frac']} of the {p['place']}?"
    ),
    answer=lambda p: p["D"] - 1,
    lure=lambda p: p["D"] / 2,
    spec=lambda p: ProblemSpec(
        variables=[
            Variable(name="day", description="day the target coverage is reached", unit="days")
        ],
        equations=[f"{p['k']} ^ day = {p['k']} ^ {p['D']} / {p['k']}"],
        query="day",
        unit="days",
        assumptions=[
            "coverage is proportional to k**day, so full coverage on day D fixes the constant"
        ],
    ),
    sample=lambda rng: _sample_lily(rng),
    surface=("thing", "place"),
)


def _sample_lily(rng: random.Random) -> dict[str, Any]:
    k, frac = rng.choice([(2, "half"), (3, "a third"), (4, "a quarter"), (5, "a fifth")])
    thing, place = rng.choice(
        [
            ("patch of algae", "reservoir"),
            ("colony of mould", "petri dish"),
            ("bloom of duckweed", "pond"),
            ("cluster of ivy", "courtyard wall"),
        ]
    )
    D = rng.randrange(9, 61)
    if D % 2:  # keep the lure a whole number so it is unambiguous to detect
        D += 1
    return {"k": k, "D": D, "frac": frac, "thing": thing, "place": place}


# --------------------------------------------------------------------------------------
# 4. Percentage round trip
# --------------------------------------------------------------------------------------
# Down p% then up p% is a net loss of p**2.  Lure: back to the start.

DISCOUNT = Template(
    family="discount",
    unit="dollars",
    trap="Equal-and-opposite percentages feel like they cancel; they apply to different bases.",
    canonical={"price": 80.0, "pct": 25, "thing": "jacket"},
    render=lambda p: (
        f"A {p['thing']} is priced at ${_money(p['price'])}. In a sale its price is cut by "
        f"{p['pct']}%. The following week the sale ends and the new price is raised by "
        f"{p['pct']}%. What does the {p['thing']} cost now, in dollars?"
    ),
    answer=lambda p: p["price"] * (1 - p["pct"] / 100) * (1 + p["pct"] / 100),
    lure=lambda p: p["price"],
    spec=lambda p: ProblemSpec(
        variables=[
            Variable(name="sale_price", description="price during the sale", unit="dollars"),
            Variable(name="final_price", description="price after the rise", unit="dollars"),
        ],
        equations=[
            f"sale_price = {p['price']} * (1 - {p['pct']} / 100)",
            f"final_price = sale_price * (1 + {p['pct']} / 100)",
        ],
        query="final_price",
        unit="dollars",
        assumptions=["the second percentage applies to the reduced price, not the original"],
    ),
    sample=lambda rng: _sample_discount(rng),
    surface=("thing",),
)


def _sample_discount(rng: random.Random) -> dict[str, Any]:
    thing = rng.choice(["jacket", "monitor", "mattress", "guitar", "office chair", "tent"])
    return {
        "price": float(rng.randrange(40, 400, 5)),
        "pct": rng.choice([10, 20, 25, 30, 40, 50]),
        "thing": thing,
    }


# --------------------------------------------------------------------------------------
# 5. Double-counted rank
# --------------------------------------------------------------------------------------
# nth highest and mth lowest -> (n-1) above + (m-1) below + the person = n + m - 1.
# Lure: n + m, double-counting the person themselves.

RANK = Template(
    family="rank",
    unit="people",
    trap="The person being described gets counted once from each end.",
    canonical={"n": 15, "m": 15, "name": "Jamie", "group": "class", "unit_noun": "students"},
    render=lambda p: (
        f"In a {p['group']}, {p['name']} finished with both the {_ordinal(p['n'])} highest score "
        f"and the {_ordinal(p['m'])} lowest score. Assuming no ties, how many {p['unit_noun']} "
        f"are in the {p['group']}?"
    ),
    answer=lambda p: p["n"] + p["m"] - 1,
    lure=lambda p: p["n"] + p["m"],
    spec=lambda p: ProblemSpec(
        variables=[
            Variable(name="above", description="people scoring higher", unit="people"),
            Variable(name="below", description="people scoring lower", unit="people"),
            Variable(name="total", description="people in the group", unit="people"),
        ],
        equations=[
            f"above = {p['n']} - 1",
            f"below = {p['m']} - 1",
            "total = above + below + 1",
        ],
        query="total",
        unit="people",
        assumptions=["the person described is a member of the group and is counted once"],
    ),
    sample=lambda rng: _sample_rank(rng),
    surface=("name", "group", "unit_noun"),
)


def _sample_rank(rng: random.Random) -> dict[str, Any]:
    name = rng.choice(["Jamie", "Priya", "Noor", "Devin", "Rosa", "Kenji", "Ada"])
    group, unit_noun = rng.choice(
        [("class", "students"), ("cohort", "trainees"), ("league", "players"), ("ward", "nurses")]
    )
    n = rng.randrange(4, 30)
    m = rng.randrange(4, 30)
    return {"n": n, "m": m, "name": name, "group": group, "unit_noun": unit_noun}


# --------------------------------------------------------------------------------------
# 6. Waitlist drift  (novel family — not in the classic CRT literature)
# --------------------------------------------------------------------------------------
# A queue drains at (served - arriving) per day, not at (served) per day.
# Lure: divide by the service rate and ignore the inflow.

DRIFT = Template(
    family="drift",
    unit="days",
    trap="The inflow is stated but feels like background; the drain rate is the salient number.",
    canonical={"backlog": 210, "served": 12, "arriving": 5, "place": "clinic"},
    render=lambda p: (
        f"A {p['place']} has {p['backlog']} people on its waitlist. It can see {p['served']} new "
        f"people per day, but {p['arriving']} new people join the waitlist each day. Starting "
        f"today, how many days until the waitlist is empty?"
    ),
    answer=lambda p: p["backlog"] / (p["served"] - p["arriving"]),
    lure=lambda p: p["backlog"] / p["served"],
    spec=lambda p: ProblemSpec(
        variables=[
            Variable(name="net_rate", description="net reduction per day", unit="people per day"),
            Variable(name="days", description="days until the waitlist is empty", unit="days"),
        ],
        equations=[
            f"net_rate = {p['served']} - {p['arriving']}",
            f"days * net_rate = {p['backlog']}",
        ],
        query="days",
        unit="days",
        assumptions=["arrivals and appointments both continue at a constant daily rate"],
    ),
    sample=lambda rng: _sample_drift(rng),
    surface=("place",),
)


def _sample_drift(rng: random.Random) -> dict[str, Any]:
    place = rng.choice(["clinic", "counselling service", "helpline", "assessment centre"])
    arriving = rng.randrange(2, 10)
    net = rng.choice([2, 3, 4, 5, 6, 7])
    served = arriving + net
    backlog = net * rng.randrange(8, 60)
    return {"backlog": backlog, "served": served, "arriving": arriving, "place": place}


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


TEMPLATES: dict[ItemFamily, Template] = {
    t.family: t for t in (BAT_BALL, WIDGETS, LILY_PAD, DISCOUNT, RANK, DRIFT)
}

#: The three families that appear verbatim in Frederick (2005) and therefore
#: saturate every pretraining corpus. Used to split "recall" from "reason".
CLASSIC_FAMILIES: tuple[ItemFamily, ...] = ("bat_ball", "widgets", "lily_pad")

#: Families we wrote ourselves. Same trap structure, no canonical wording to recall.
NOVEL_FAMILIES: tuple[ItemFamily, ...] = ("discount", "rank", "drift")
