import json

import numpy as np
import pytest

from cheeseblend.agent import quote_supports, verify
from cheeseblend.catalogue import AROMA, DATA, SCORED, Cheese, Compound, build, load_cheeses, to_canonical
from cheeseblend.engine import (COL, NA, P, Target, cosine, describe, fingerprint, mix_everything, optimize,
                                predict)


def cheese(id, salt=1.5, ph=5.4, filata=False, non_melter=False, compounds=(), **scores):
    comp = {"moisture_pct": 40, "fat_pct": 30, "salt_pct": salt, "ph": ph}
    return Cheese(
        id=id, name=id.title(), milk="cow", treatment="pasteurized", style="semi_hard",
        composition={k: {"value": v} for k, v in comp.items()},
        scores={d: {"score": scores.get(d, 0.0)} for d in SCORED},
        filata=filata, non_melter=non_melter, compounds=list(compounds),
    )


CHEESES = [
    cheese("mozz", filata=True, melt=8, stretch=10, buttery=3, sour_vinegar=2, creaminess=6, salt=1.0, ph=5.3),
    cheese("alpine", melt=8, stretch=4, malty_nutty=8, sulfur_cooked=4, fruity=3, umami=6, browning=5),
    cheese("blue", melt=6, blue_moldy=9, cheesy_sweaty=6, salt=3.5, ph=6.2),
    cheese("halloumi", non_melter=True, melt=1, firmness=7, salt=2.8, buttery=2),
    cheese("grana", melt=3, malty_nutty=7, cheesy_sweaty=5, umami=10, crumbliness=9, salt=1.6),
]
SPACE = json.loads((DATA / "odor_space.json").read_text("utf-8"))
CAT = build(CHEESES, {}, [{"name": "pizza", "cheeses": {"mozz": 0.7, "alpine": 0.3}}], SPACE)
I = CAT.index
TARGET = Target(dims={"melt": {"value": 9, "importance": 5}, "stretch": {"value": 8, "importance": 4},
                      "malty_nutty": {"value": 6, "importance": 3}})


@pytest.mark.parametrize("c", CHEESES, ids=lambda c: c.id)
def test_solo_blend_returns_own_scores(c):
    prof = predict(CAT, [I[c.id]], [[1.0]])[0]
    for d in SCORED:
        assert prof[COL[d]] == pytest.approx(c.scores[d].score, abs=1e-9), d
    assert prof[COL["salt"]] == pytest.approx(c.composition["salt_pct"].value * 2)


def test_mixing_a_cheese_with_itself_changes_nothing():
    i = I["alpine"]
    assert np.allclose(predict(CAT, [i, i], [[0.3, 0.7]]), predict(CAT, [i], [[1.0]]))


def test_masking_leaves_strongest_family_untouched():
    idx, w = [I["alpine"], I["blue"]], np.array([[0.5, 0.5]])
    intensity = np.clip(w @ CAT.oav[idx], 1, None) ** P.n - 1
    prof = predict(CAT, idx, w)[0]
    assert prof[:NA].max() == pytest.approx((10 * intensity / CAT.i_ref).max())


def test_more_blue_never_lowers_blue():
    shares = np.linspace(0.1, 0.9, 17)
    blue = predict(CAT, [I["blue"], I["mozz"]], np.c_[shares, 1 - shares])[:, COL["blue_moldy"]]
    assert np.all(np.diff(blue) >= -1e-12)


def test_stretch_survives_dilution_but_needs_enough_filata():
    shares = np.array([[0.5, 0.5], [0.1, 0.9]])
    st = predict(CAT, [I["mozz"], I["grana"]], shares)[:, COL["stretch"]]
    assert st[0] > 9 and st[1] < 2


def test_non_melter_breaks_melt():
    melt = predict(CAT, [I["mozz"], I["halloumi"]], [[0.5, 0.5]])[0, COL["melt"]]
    assert melt < 0.5 * 8 + 0.5 * 1


def test_optimize_shares_valid_and_deterministic():
    a, b = optimize(CAT, TARGET), optimize(CAT, TARGET)
    assert a == b and a
    for r in a:
        shares = np.array(list(r["shares"].values()))
        assert shares.sum() == pytest.approx(1, abs=1e-6)
        assert shares.min() >= P.min_share - 1e-6 and len(shares) <= P.k_max
    assert len({next(iter(r["shares"])) for r in a}) == len(a)  # distinct lead cheeses


def test_optimized_beats_mix_everything():
    assert optimize(CAT, TARGET)[0]["score"] > mix_everything(CAT, TARGET)["score"]


def test_exclusions():
    t = TARGET.model_copy(update={"exclude": ["mozz"], "exclude_families": ["blue_moldy"]})
    for r in optimize(CAT, t):
        assert "mozz" not in r["shares"] and "blue" not in r["shares"]


def test_measured_compounds_drive_aroma():
    def lab(id, conc):
        return cheese(id, compounds=[Compound(name="diacetyl", value=conc, unit="µg/kg", confidence="measured")],
                      buttery=2, malty_nutty=5)
    cat = build([lab("lo", 0.1), lab("hi", 0.4), *CHEESES], {"diacetyl": 0.001}, [], SPACE)  # OAV 100 vs 400
    assert cat.measured[0, AROMA.index("buttery")] and not cat.measured[0, AROMA.index("malty_nutty")]
    lo, hi = predict(cat, [0, 1], np.eye(2))[:, COL["buttery"]]
    assert hi > 2 > lo  # same Muse score, chemistry splits them around it
    assert cat.conf[0] > cat.conf[2]


def test_fingerprint_moves_toward_the_cheese_you_add_more_of():
    idx = [I["blue"], I["mozz"]]
    blue = fingerprint(CAT, [I["blue"]], [[1.0]], predict(CAT, [I["blue"]], [[1.0]]))[0]
    W = np.c_[np.linspace(0.1, 0.9, 9), np.linspace(0.9, 0.1, 9)]
    sims = cosine(fingerprint(CAT, idx, W, predict(CAT, idx, W)), blue)
    assert np.all(np.diff(sims) > 0)
    i = I["alpine"]
    assert np.allclose(fingerprint(CAT, [i, i], [[0.4, 0.6]], predict(CAT, [i, i], [[0.4, 0.6]])),
                       fingerprint(CAT, [i], [[1.0]], predict(CAT, [i], [[1.0]])))


def test_character_prefers_the_cheese_that_smells_like_the_target():
    nutty = Target(mode="cold", dims={"malty_nutty": {"value": 8, "importance": 5}})
    assert describe(CAT, [I["alpine"]], [1.0], nutty)["character"] > describe(CAT, [I["blue"]], [1.0], nutty)["character"]


def test_holdout_removes_recipe_pairs():
    blends, rec = [{"name": "pizza", "cheeses": {"mozz": 0.7, "alpine": 0.3}}], {"pairs": {"alpine|mozz": 0.4}}
    a, b = I["mozz"], I["alpine"]
    kept = build(CHEESES, {}, [], SPACE, rec).harmony[a, b]
    held = build(CHEESES, {}, blends, SPACE, rec, holdout="pizza").harmony[a, b]
    assert kept - held == pytest.approx(0.5)  # 0.5 x normalized recipe NPMI of 1


def test_describe_orders_by_share():
    r = describe(CAT, [I["mozz"], I["alpine"]], [0.3, 0.7], TARGET)
    assert list(r["shares"]) == ["alpine", "mozz"]


def test_catalogue_files_validate():
    load_cheeses()  # raises on any invalid file


def test_to_canonical_handles_real_extractor_units():
    assert to_canonical("diacetyl", 2.3, "ppb in water") == 2.3
    assert to_canonical("acetic acid", 99, "mg/100g") == 990_000
    assert to_canonical("acetic acid", 100, "mg") is None  # no denominator
    assert to_canonical("moisture_pct", 29.2, "w/w") == 29.2
    assert to_canonical("salt_pct", 2.26, "DM centre") is None  # dry-matter basis
    assert to_canonical("fat_pct", 306, "g/kg") == pytest.approx(30.6)
    assert to_canonical("salt_pct", 27.0, "g kg-1") == pytest.approx(2.7)
    assert to_canonical("butanoic acid", 4.2, "mg kg−1") == pytest.approx(4200)
    assert to_canonical("salt_pct", 1.6, "g.100g-1") == pytest.approx(1.6)


def test_quote_supports():
    assert quote_supports(1234.5, "found 1,234.5 µg/kg")
    assert quote_supports(12.5, "valeur de 12,5 mg/kg")
    assert not quote_supports(12.0, "found 21 µg/kg")


def test_verify_requires_quote_in_notes_and_cited_url():
    notes = {"text": "Diacetyl was 120 µg/kg (https://a.org/p).", "citations": [{"url": "https://a.org/p"}]}
    good = {"value": 120, "quote": "Diacetyl was 120 µg/kg", "source_url": "https://a.org/p"}
    assert verify(good, notes)
    assert not verify({**good, "quote": "Diacetyl was 120 µg/kg in aged samples"}, notes)
    assert not verify({**good, "source_url": "https://made.up/x"}, notes)
