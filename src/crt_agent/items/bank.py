"""Construction of the three item sets the benchmark compares.

    canonical  — the literal, famous wordings. Every model has read these thousands
                 of times. High scores here mean nothing on their own.
    surface    — the canonical *numbers*, different nouns. The answer is unchanged;
                 only the wording moved.
    perturbed  — the canonical wording structure, different numbers. Recall of the
                 answer cannot help.
    novel      — families we invented. Neither wording nor number is recallable.

`surface` and `perturbed` exist as separate sets because collapsing them confounds
two different failure modes. If an agent drops on `surface` but holds on `perturbed`,
it was keyed to the phrasing. If it holds on `surface` but drops on `perturbed`, it
was reciting the number. Perturbing both at once cannot tell you which happened, and
that is the first question anyone will ask about the result.

The interesting quantity is never accuracy on one set — it is the gaps between them.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from crt_agent.items.templates import (
    CLASSIC_FAMILIES,
    NOVEL_FAMILIES,
    TEMPLATES,
)
from crt_agent.schemas import Item, ItemFamily, ItemSet


def _build(family: ItemFamily, params: dict, item_set: ItemSet, suffix: str) -> Item:
    tpl = TEMPLATES[family]
    return Item(
        item_id=f"{family}:{item_set}:{suffix}",
        family=family,
        item_set=item_set,
        text=tpl.render(params),
        answer=round(float(tpl.answer(params)), 10),
        lure=round(float(tpl.lure(params)), 10),
        unit=tpl.unit,
        tolerance=1e-4,
        reference_spec=tpl.spec(params),
    )


def canonical_items() -> list[Item]:
    """The famous three, verbatim, plus the canonical form of our own families."""
    out = []
    for family in CLASSIC_FAMILIES:
        out.append(_build(family, TEMPLATES[family].canonical, "canonical", "0"))
    return out


def novel_items(n_per_family: int = 4, seed: int = 20260825) -> list[Item]:
    """Families with no canonical wording in the literature."""
    rng = random.Random(seed)
    out = []
    for family in NOVEL_FAMILIES:
        out.append(_build(family, TEMPLATES[family].canonical, "novel", "0"))
        for i in range(1, n_per_family):
            out.append(_build(family, TEMPLATES[family].sample(rng), "novel", str(i)))
    return out


def surface_items(
    n_per_family: int = 6,
    seed: int = 20260825,
    families: tuple[ItemFamily, ...] = CLASSIC_FAMILIES,
) -> list[Item]:
    """Canonical numbers, resampled nouns.

    Every one of these has exactly the same correct answer as its canonical parent —
    0.05, 5, 47. Only the wording changed. An agent that recalls the *answer* should
    score as well here as on the canonical set; an agent that recalls the *wording*
    should not.
    """
    rng = random.Random(seed ^ 0x5F5F)
    out: list[Item] = []
    for family in families:
        tpl = TEMPLATES[family]
        seen: set[str] = set()
        made, attempts = 0, 0
        while made < n_per_family and attempts < n_per_family * 200:
            attempts += 1
            params = dict(tpl.canonical)
            drawn = tpl.sample(rng)
            params.update({k: drawn[k] for k in tpl.surface if k in drawn})
            text = tpl.render(params)
            if text in seen or text == tpl.render(tpl.canonical):
                continue
            seen.add(text)
            item = _build(family, params, "surface", str(made))
            # Guard the invariant this set exists to provide.
            canonical_answer = float(tpl.answer(tpl.canonical))
            if abs(item.answer - canonical_answer) > 1e-9:
                raise RuntimeError(f"{family}: surface variant changed the answer")
            out.append(item)
            made += 1
    return out


def perturbed_items(
    n_per_family: int = 10,
    seed: int = 20260825,
    families: tuple[ItemFamily, ...] = CLASSIC_FAMILIES,
) -> list[Item]:
    """Resample the classics.

    Rejects any draw where the correct answer coincides with the lure (the trap has
    to still be a trap) or where the answer is not cleanly representable.
    """
    rng = random.Random(seed)
    out: list[Item] = []
    for family in families:
        tpl = TEMPLATES[family]
        made, attempts = 0, 0
        while made < n_per_family and attempts < n_per_family * 200:
            attempts += 1
            params = tpl.sample(rng)
            answer, lure = float(tpl.answer(params)), float(tpl.lure(params))
            if abs(answer - lure) < 1e-3 or answer <= 0:
                continue
            item = _build(family, params, "perturbed", str(made))
            out.append(item)
            made += 1
        if made < n_per_family:  # pragma: no cover - sampler is comfortably wide
            raise RuntimeError(f"could not sample {n_per_family} valid items for {family}")
    return out


def default_suite(
    n_perturbed: int = 10,
    n_novel: int = 4,
    seed: int = 20260825,
    n_surface: int | None = None,
) -> list[Item]:
    """Canonical + surface + perturbed + novel, in that order."""
    return [
        *canonical_items(),
        *surface_items(n_perturbed if n_surface is None else n_surface, seed=seed),
        *perturbed_items(n_perturbed, seed=seed),
        *novel_items(n_novel, seed=seed),
    ]


def iter_sets(items: list[Item]) -> Iterator[tuple[ItemSet, list[Item]]]:
    for name in ("canonical", "surface", "perturbed", "novel"):
        subset = [i for i in items if i.item_set == name]
        if subset:
            yield name, subset  # type: ignore[misc]
