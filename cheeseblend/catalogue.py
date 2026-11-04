"""Cheese schema, compound families, and the catalogue as numpy arrays for the engine."""

import json
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

DATA = Path(__file__).resolve().parent.parent / "data"

FAMILIES = {
    "buttery": ["diacetyl", "acetoin", "delta-decalactone", "delta-dodecalactone"],
    "cheesy_sweaty": ["butanoic acid", "3-methylbutanoic acid", "hexanoic acid", "octanoic acid"],
    "sour_vinegar": ["acetic acid", "propanoic acid"],
    "goaty": ["4-methyloctanoic acid", "4-ethyloctanoic acid"],
    "malty_nutty": ["3-methylbutanal", "2-methylbutanal", "2-methylpropanal"],
    "sulfur_cooked": ["methanethiol", "dimethyl sulfide", "dimethyl disulfide", "dimethyl trisulfide", "methional"],
    "blue_moldy": ["2-heptanone", "2-nonanone", "2-undecanone"],
    "fruity": ["ethyl butanoate", "ethyl hexanoate"],
    "mushroom_earthy": ["1-octen-3-ol", "1-octen-3-one"],
    "floral_honey": ["phenylacetaldehyde", "2-phenylethanol"],
    "animal_barnyard": ["p-cresol", "indole"],
    "smoky": ["guaiacol"],
    "ammonia": ["ammonia"],
}
COMPOUNDS = [c for cs in FAMILIES.values() for c in cs]
FAMILY_OF = {c: f for f, cs in FAMILIES.items() for c in cs}

AROMA = list(FAMILIES)
FAM_IDX = np.array([AROMA.index(FAMILY_OF[c]) for c in COMPOUNDS])
MEMBER = np.eye(len(AROMA))[FAM_IDX]  # (compounds, families) one-hot
TASTE = ["salt", "sour", "umami", "bitter", "sweet"]
TEXTURE = ["firmness", "creaminess", "crumbliness"]
FUNCTION = ["melt", "stretch", "browning", "oil_off"]
DIMS = AROMA + TASTE + TEXTURE + FUNCTION
SCORED = AROMA + ["umami", "bitter", "sweet"] + TEXTURE + FUNCTION  # 0-10 from the Muse gap filler
LINEAR = ["salt", "umami", "bitter", "sweet"] + TEXTURE + ["browning", "oil_off"]  # mix by mass balance
COMPOSITION = ["moisture_pct", "fat_pct", "salt_pct", "ph"]

PH_FLAT, PH_SOUR = 6.8, 4.3  # pH read as sour 0 and sour 10
SALT_PER_PCT = 2.0  # 5 % NaCl reads as salt 10
I_REF_DEFAULT = 10.0  # intensity that reads as 10 when nothing is measured
MEASURED_WEIGHT = 0.5  # log-scale weight of measured chemistry vs the Muse estimate in a measured family

# Canonical units: µg/kg for compounds and thresholds (in water µg/L ≈ µg/kg), % for composition.
MASS = {"µg/kg": 1, "ug/kg": 1, "ng/g": 1, "ppb": 1, "µg/l": 1, "ug/l": 1, "mg/kg": 1e3, "µg/g": 1e3, "ug/g": 1e3,
        "ppm": 1e3, "mg/l": 1e3, "ng/kg": 1e-3, "ng/l": 1e-3, "mg/100g": 1e4, "mg/100 g": 1e4, "g/kg": 1e6}
MASS_KEYS = sorted(MASS, key=len, reverse=True)  # longest prefix first
PCT = {"g/kg": 0.1, "mg/100g": 1e-3, "mg/100 g": 1e-3, "mg/kg": 1e-4}  # anything else is read as %


def to_canonical(name: str, value: float, unit: str) -> float | None:
    """None when the unit can't be converted (e.g. relative peak area, dry-matter basis): the value is dropped.
    Units come with qualifiers ("ppb in water", "% w/w centre"), so they match by prefix."""
    u = unit.strip().lower().replace("μ", "µ")  # Greek mu vs micro sign
    u = re.sub(r"\s*[·.]?\s*(100\s*g|kg|g|ml|l)\s*(?:-1|−1|⁻¹)", r"/\1", u)  # "mg kg-1", "g.100g-1"
    u = re.sub(r"\s*/\s*", "/", u)
    if name == "ph":
        return value if 3 < value < 8 else None
    if name in COMPOSITION:
        # ponytail: research asks for %, so anything without a mass unit is read as %; build cross-checks vs Muse
        if "dm" in u.replace("/", " ").split() or "dry" in u:
            return None
        v = value * next((PCT[k] for k in PCT if u.startswith(k)), 1)
        return v if 0 < v < 100 else None
    f = next((MASS[k] for k in MASS_KEYS if u.startswith(k)), None)
    return None if f is None else value * f


class Measurement(BaseModel):
    value: float
    unit: str = ""
    source_url: str = ""
    quote: str = ""
    confidence: Literal["measured", "inferred"] = "inferred"


class Compound(Measurement):
    name: str


class Score(BaseModel):
    score: float = Field(ge=0, le=10)
    reason: str = ""


class Cheese(BaseModel):
    id: str
    name: str
    milk: str
    treatment: str
    style: str
    origin: str = ""
    process: dict = {}
    composition: dict[str, Measurement]
    compounds: list[Compound] = []
    scores: dict[str, Score]
    filata: bool = False
    non_melter: bool = False
    sources: list[str] = []

    @model_validator(mode="after")
    def _complete(self):
        missing = (set(SCORED) - set(self.scores)) | (set(COMPOSITION) - set(self.composition))
        if missing:
            raise ValueError(f"{self.id}: missing {sorted(missing)}")
        return self


@dataclass
class Catalogue:
    cheeses: list[Cheese]
    oav: np.ndarray  # (N, families) odor activity: measured where we have it, else back-solved from scores
    measured: np.ndarray  # (N, families) bool
    lin: np.ndarray  # (N, LINEAR)
    ph: np.ndarray
    melt: np.ndarray
    stretch: np.ndarray
    filata: np.ndarray
    non_melter: np.ndarray
    conf: np.ndarray  # 0-1, share of aroma + composition values that are measured
    i_ref: np.ndarray  # (families,) perceived intensity that reads as 10
    harmony: np.ndarray  # (N, N) h_ab, zero diagonal
    coav: np.ndarray  # (N, compounds) odor activity per compound; sums to oav within each family
    emb: np.ndarray  # (compounds, D) unit odor embedding per compound
    fam_vec: np.ndarray  # (families, D) unit embedding of a typical family member mix
    white: np.ndarray  # (D,) "cheese white": the catalogue-average odor vector that big mixes converge to
    descriptors: dict[str, np.ndarray]  # odor word -> unit embedding, same space

    @property
    def index(self) -> dict[str, int]:
        return {c.id: i for i, c in enumerate(self.cheeses)}


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def build(cheeses: list[Cheese], thresholds: dict[str, float], blends: list[dict], space: dict,
          recipes: dict | None = None, n: float = 0.5, m: float = 0.5, holdout: str | None = None) -> Catalogue:
    """n, m must match the engine Params. space: data/odor_space.json. recipes: data/recipe_pairs.json.
    holdout: blend name whose pairs are left out of co-occurrence and recipe data (backtest)."""
    N, F = len(cheeses), len(AROMA)
    c_oav = np.zeros((N, len(COMPOUNDS)))
    for i, c in enumerate(cheeses):
        for x in c.compounds:
            if x.confidence == "measured" and thresholds.get(x.name) and x.name in FAMILY_OF:
                c_oav[i, COMPOUNDS.index(x.name)] += x.value / thresholds[x.name]
    m_oav, measured = c_oav @ MEMBER, (c_oav > 0) @ MEMBER > 0

    # Gap-filler scores are as perceived, i.e. after masking. Undo masking so a solo cheese predicts its own scores.
    s = np.array([[c.scores[a].score for a in AROMA] for c in cheeses]).reshape(N, F)
    j_inf = (s * s.max(1, keepdims=True) ** m) ** (1 / (1 + m))  # pre-masking intensity on the 0-10 scale

    # Thresholds span 6 orders of magnitude, so each family gets its own scale: the median ratio of measured
    # intensity to the gap filler's intensity over cheeses that have both. Chemistry orders cheeses within a family,
    # the sensory scale anchors it. ponytail: median over few cheeses (2-18 per family); fit jointly if it wobbles.
    i_meas = np.clip(m_oav, 1, None) ** n - 1
    pair = measured & (j_inf >= 1) & (i_meas > 0)
    ratio = np.where(pair, 10 * i_meas / np.maximum(j_inf, 1e-12), np.nan)
    fallback = float(np.nanmedian(ratio)) if pair.any() else I_REF_DEFAULT
    i_ref = np.array([np.nanmedian(ratio[:, k]) if pair[:, k].any() else fallback for k in range(F)])
    # Measured and Muse intensities are two noisy estimates of one thing (extraction methods across papers disagree by
    # orders of magnitude; rank correlation ~0.4), so fuse them on a log scale instead of letting either override.
    j_meas = 10 * i_meas / i_ref
    w = MEASURED_WEIGHT
    j = np.where(measured, np.expm1(w * np.log1p(j_meas) + (1 - w) * np.log1p(j_inf)), j_inf)
    oav = (j * i_ref / 10 + 1) ** (1 / n)

    lin = np.array([[c.composition["salt_pct"].value * SALT_PER_PCT if d == "salt" else c.scores[d].score
                     for d in LINEAR] for c in cheeses]).reshape(N, len(LINEAR))
    comp_measured = np.array([[c.composition[k].confidence == "measured" for k in COMPOSITION]
                              for c in cheeses]).reshape(N, len(COMPOSITION))
    conf = 0.3 + 0.7 * (measured.sum(1) + comp_measured.sum(1)) / (F + len(COMPOSITION))

    # Split each family's odor activity over its compounds: measured proportions where measured, else the typical
    # proportions of the cheeses that were measured (equal split for families nobody measured).
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.nan_to_num(c_oav / (m_oav @ MEMBER.T))
    fam_measured = measured @ MEMBER.T > 0  # (N, compounds)
    counts = fam_measured.sum(0)
    typical = np.where(counts > 0, share.sum(0) / np.maximum(counts, 1), 1 / MEMBER.sum(0)[FAM_IDX])
    typical /= (typical @ MEMBER) @ MEMBER.T
    cshare = np.where(fam_measured, share, typical)
    coav = oav @ MEMBER.T * cshare

    emb = np.array([space["compounds"][c]["vec"] for c in COMPOUNDS])
    fam_vec = _unit(MEMBER.T @ (typical[:, None] * emb))
    # Harmony: food pairing (shared compounds, Ahn et al. 2011) as the cosine of compound fingerprints
    # + recipe co-occurrence (FlavorGraph NPMI and known human blends) - clash rules.
    fp = _unit((j @ MEMBER.T * cshare) @ emb)  # pre-masking fingerprint per cheese
    white = _unit(fp.sum(0))  # where many-cheese mixes converge (Weiss et al. 2012), measured: 0.33 -> 0.08 rad
    index = {c.id: i for i, c in enumerate(cheeses)}
    held = set(next((b["cheeses"] for b in blends if b["name"] == holdout), ()))
    co = np.zeros((N, N))
    for b in blends:
        if b["name"] != holdout:
            for a, z in combinations([index[x] for x in b["cheeses"] if x in index], 2):
                co[a, z] += 1
                co[z, a] += 1
    co /= max(co.max(initial=0), 1)
    rec = np.zeros((N, N))
    for key, v in (recipes or {}).get("pairs", {}).items():
        a, z = key.split("|")
        if a in index and z in index and not {a, z} <= held:
            rec[index[a], index[z]] = rec[index[z], index[a]] = v
    rec /= max(rec.max(initial=0), 1e-12)
    strong = {f: s[:, AROMA.index(f)] >= 6 for f in ("animal_barnyard", "blue_moldy", "smoky")}
    clash = (np.outer(strong["animal_barnyard"], strong["animal_barnyard"])
             | np.outer(strong["blue_moldy"], strong["smoky"]) | np.outer(strong["smoky"], strong["blue_moldy"]))
    harmony = 0.5 * fp @ fp.T + 0.5 * np.maximum(co, rec) - 0.5 * clash
    np.fill_diagonal(harmony, 0)

    def col(key):
        return np.array([c.scores[key].score for c in cheeses])

    return Catalogue(
        cheeses=cheeses, oav=oav, measured=measured, lin=lin,
        ph=np.array([c.composition["ph"].value for c in cheeses]),
        melt=col("melt"), stretch=col("stretch"),
        filata=np.array([c.filata for c in cheeses], float),
        non_melter=np.array([c.non_melter for c in cheeses], float),
        conf=conf, i_ref=i_ref, harmony=harmony, coav=coav, emb=emb, fam_vec=fam_vec, white=white,
        descriptors={k: np.array(v) for k, v in space["descriptors"].items()},
    )


def _json(name: str, default):
    path = DATA / name
    return json.loads(path.read_text("utf-8")) if path.exists() else default


def load_cheeses() -> list[Cheese]:
    return [Cheese.model_validate_json(p.read_text("utf-8")) for p in sorted((DATA / "catalogue").glob("*.json"))]


def load(holdout: str | None = None, n: float = 0.5, m: float = 0.5) -> Catalogue:
    thresholds = {k: v["value"] for k, v in _json("thresholds.json", {}).items()}
    return build(load_cheeses(), thresholds, _json("blends.json", []), _json("odor_space.json", None),
                 _json("recipe_pairs.json", {}), n, m, holdout)
