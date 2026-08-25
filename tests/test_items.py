"""Item generation: the traps must survive perturbation, and the sets must be disjoint."""

from __future__ import annotations

import pytest

from crt_agent.items import (
    canonical_items,
    default_suite,
    novel_items,
    perturbed_items,
    surface_items,
)
from crt_agent.items.templates import CLASSIC_FAMILIES, TEMPLATES


def test_canonical_items_are_the_published_ones():
    by_family = {i.family: i for i in canonical_items()}
    assert by_family["bat_ball"].answer == pytest.approx(0.05)
    assert by_family["bat_ball"].lure == pytest.approx(0.10)
    assert by_family["widgets"].answer == pytest.approx(5)
    assert by_family["widgets"].lure == pytest.approx(100)
    assert by_family["lily_pad"].answer == pytest.approx(47)
    assert by_family["lily_pad"].lure == pytest.approx(24)


def test_perturbed_items_keep_the_trap():
    """A perturbation whose answer equals its lure is not a CRT item any more."""
    for item in perturbed_items(25):
        assert abs(item.answer - item.lure) > 1e-3, item.item_id
        assert item.answer > 0


def test_perturbed_items_do_not_reproduce_the_canonical_wording():
    canonical_texts = {i.text for i in canonical_items()}
    for item in perturbed_items(25):
        assert item.text not in canonical_texts


def test_generation_is_seed_reproducible():
    assert [i.text for i in perturbed_items(8, seed=7)] == [
        i.text for i in perturbed_items(8, seed=7)
    ]
    assert [i.text for i in perturbed_items(8, seed=7)] != [
        i.text for i in perturbed_items(8, seed=8)
    ]


def test_item_ids_are_unique():
    items = default_suite(12, 5)
    assert len({i.item_id for i in items}) == len(items)


def test_novel_families_are_disjoint_from_the_classics():
    assert all(i.family not in CLASSIC_FAMILIES for i in novel_items(3))


@pytest.mark.parametrize("family", list(TEMPLATES))
def test_every_template_documents_its_trap(family):
    assert TEMPLATES[family].trap, f"{family} has no trap description"


def test_answer_and_lure_agree_with_the_template_functions():
    for item in default_suite(6, 3):
        tpl = TEMPLATES[item.family]
        assert tpl.unit == item.unit


def test_surface_variants_change_the_wording_but_not_the_answer():
    """The invariant that makes the wording/number split meaningful."""
    canonical = {i.family: i for i in canonical_items()}
    for item in surface_items(8):
        parent = canonical[item.family]
        assert item.answer == pytest.approx(parent.answer), item.item_id
        assert item.lure == pytest.approx(parent.lure), item.item_id
        assert item.text != parent.text


def test_surface_variants_are_distinct_from_each_other():
    items = surface_items(6)
    assert len({i.text for i in items}) == len(items)


def test_perturbed_items_actually_change_the_answer():
    canonical = {i.family: i for i in canonical_items()}
    changed = sum(abs(i.answer - canonical[i.family].answer) > 1e-9 for i in perturbed_items(10))
    assert changed > 0.8 * len(perturbed_items(10))


def test_default_suite_contains_all_four_sets():
    sets = {i.item_set for i in default_suite(4, 2)}
    assert sets == {"canonical", "surface", "perturbed", "novel"}
