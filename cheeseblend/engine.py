"""Blend shares -> predicted profile -> hedonic score -> search. Pure numpy, never calls Muse.
Everything is batched: W is (B, k) shares over the cheeses in idx, so a whole share grid scores in one call."""

import itertools
from dataclasses import dataclass
from functools import cache

import numpy as np
from pydantic import BaseModel
from scipy.optimize import minimize
from scipy.special import expit

from .catalogue import AROMA, COMPOUNDS, DIMS, FUNCTION, LINEAR, MEMBER, PH_FLAT, PH_SOUR, Catalogue

COL = {d: i for i, d in enumerate(DIMS)}
LIN_COLS = [COL[d] for d in LINEAR]
NA = len(AROMA)


@dataclass(frozen=True)
class Params:
    n: float = 0.5  # Stevens exponent (rebuild the catalogue if you change n or m)
    m: float = 0.5  # masking strength
    e_star: float = 3.5  # Wundt peak, in effective aroma families
    sigma_e: float = 1.5
    beta: float = 0.3  # Wundt weight
    gamma: float = 0.2  # harmony weight
    lam: float = 0.02  # penalty per extra cheese
    mu: float = 0.1  # uncertainty penalty
    sigma: float = 1.5  # bliss width per dimension
    melt_penalty: float = 4.0  # a non-melter breaks the melt of the cheese around it
    theta: float = 0.3  # pasta filata share where stretch switches on
    s: float = 0.08  # sharpness of that switch
    aroma_floor: float = 0.5  # families below this don't count as present (re-tuned on train with 171 cheeses: ties best)
    exclude_level: float = 4.0  # a cheese "has" an excluded family at this score
    min_share: float = 0.10
    k_max: int = 4
    beam: int = 10
    candidates: int = 30
    step: float = 0.05  # share grid; finalists are polished with SLSQP


P = Params()


class Dim(BaseModel):
    value: float
    importance: float = 3


class Target(BaseModel):
    mode: str = "cooked"
    dims: dict[str, Dim]
    milk: list[str] = []
    exclude_families: list[str] = []
    exclude: list[str] = []  # cheese ids


def predict(cat: Catalogue, idx: list[int], W, p: Params = P) -> np.ndarray:
    """(B, k) shares -> (B, len(DIMS)) predicted 0-10 profile."""
    W = np.atleast_2d(np.asarray(W, float))
    I = np.clip(W @ cat.oav[idx], 1, None) ** p.n - 1  # mass balance on OAV, then Stevens: 0 below threshold
    J = 10 * I / cat.i_ref  # per-family scale: 10 = the most intense cheese
    F = J * (J / np.maximum(J.max(1, keepdims=True), 1e-12)) ** p.m  # masking: strongest family untouched
    ph = -np.log10(W @ 10.0 ** -cat.ph[idx])  # ignores protein buffering
    wnm = W @ cat.non_melter[idx]
    wf = W @ cat.filata[idx]
    filata_stretch = W @ (cat.stretch * cat.filata)[idx] / np.maximum(wf, 1e-12)
    gate = expit((wf - p.theta) / p.s) / expit((1 - p.theta) / p.s)  # needs enough pasta filata; 1 when pure

    out = np.empty((len(W), len(DIMS)))
    out[:, :NA] = F
    out[:, LIN_COLS] = W @ cat.lin[idx]
    out[:, COL["sour"]] = (PH_FLAT - ph) / (PH_FLAT - PH_SOUR) * 10
    out[:, COL["melt"]] = W @ cat.melt[idx] - p.melt_penalty * wnm * (1 - wnm)  # interaction: 0 for a pure cheese
    out[:, COL["stretch"]] = np.maximum(W @ cat.stretch[idx], filata_stretch * gate)
    return np.clip(out, 0, 10)


def perceived_compounds(cat: Catalogue, idx: list[int], W, prof: np.ndarray) -> np.ndarray:
    """(B, compounds): each family's predicted intensity split by its compounds' share of the mix's odor activity.
    ponytail: linear split inside a family; per-compound Stevens if one compound should dominate less."""
    W = np.atleast_2d(np.asarray(W, float))
    mixed = W @ cat.coav[idx]
    share = mixed / np.maximum((mixed @ MEMBER) @ MEMBER.T, 1e-12)
    return prof[:, :NA] @ MEMBER.T * share


def fingerprint(cat: Catalogue, idx: list[int], W, prof: np.ndarray) -> np.ndarray:
    """(B, D) odor vector of the mix: intensity-weighted sum of compound embeddings. Mixture perception as the sum of
    component vectors, compared by angle (Snitz et al. 2013; Ravia et al. 2020)."""
    return perceived_compounds(cat, idx, W, prof) @ cat.emb


def cosine(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (X @ y) / np.maximum(np.linalg.norm(X, axis=-1) * np.linalg.norm(y), 1e-12)


def character(cat: Catalogue, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cosine after centering on the cheese white: every cheese shares a big cheesy/buttery direction, so raw cosine
    sits at 0.5-0.95 for everything. Centered, it asks whether X departs from the average cheese the way y does."""
    def unit(v):
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    return cosine(unit(X) - cat.white, unit(y) - cat.white)


def target_vector(cat: Catalogue, target: Target) -> np.ndarray:
    """The aroma the target asks for, in the same space: Σ value × typical family vector."""
    aroma = [d for d in target.dims if d in AROMA]
    return np.array([target.dims[d].value for d in aroma]) @ cat.fam_vec[[AROMA.index(d) for d in aroma]] \
        if aroma else np.zeros(cat.emb.shape[1])


def effective_families(aroma: np.ndarray, floor: float = 0.5) -> np.ndarray:
    """exp(Shannon entropy) of the perceived aroma profile: how many families are effectively present."""
    x = np.where(aroma >= floor, aroma, 0.0)
    total = x.sum(1, keepdims=True)
    q = x / np.maximum(total, 1e-12)
    return np.where(total[:, 0] > 0, np.exp(-(q * np.log(np.where(q > 0, q, 1))).sum(1)), 0.0)


def _active(target: Target) -> list[str]:
    return [d for d in target.dims if d in COL and (target.mode == "cooked" or d not in FUNCTION)]


def score(cat: Catalogue, idx: list[int], W, prof: np.ndarray, target: Target, p: Params = P):
    """H(w) = Fit + β·Wundt + γ·Harmony - λ·(n-1) - μ·Uncertainty. Returns (H, terms, E), each (B,).
    Fit: bliss point per target dimension, plus aroma character (cosine to the target's odor vector) weighted like
    the most important aroma dimension."""
    W = np.atleast_2d(np.asarray(W, float))
    dims = _active(target)
    if dims:
        v = np.array([target.dims[d].value for d in dims])
        a = np.array([target.dims[d].importance for d in dims], float)
        g = np.exp(-(prof[:, [COL[d] for d in dims]] - v) ** 2 / (2 * p.sigma**2))
        t = target_vector(cat, target)
        if np.linalg.norm(t) > 0:
            char = np.maximum(character(cat, fingerprint(cat, idx, W, prof), t), 0)
            g, a = np.c_[g, char], np.r_[a, max(target.dims[d].importance for d in dims if d in AROMA)]
        fit = g @ a / a.sum()
    else:
        fit = np.zeros(len(W))
    E = effective_families(prof[:, :NA], p.aroma_floor)
    terms = {
        "fit": fit,
        "wundt": p.beta * np.exp(-((E - p.e_star) ** 2) / (2 * p.sigma_e**2)),
        "harmony": p.gamma * 0.5 * np.einsum("bi,ij,bj->b", W, cat.harmony[np.ix_(idx, idx)], W),  # Σ_{a<b}
        "size": np.full(len(W), -p.lam * (len(idx) - 1)),
        "uncertainty": -p.mu * (W @ (1 - cat.conf[idx])),
    }
    return sum(terms.values()), terms, E


def objective(cat: Catalogue, idx: list[int], W, target: Target, p: Params = P) -> np.ndarray:
    return score(cat, idx, W, predict(cat, idx, W, p), target, p)[0]


@cache
def simplex_grid(k: int, min_share: float, step: float) -> np.ndarray:
    """Every share vector on a `step` grid with Σw = 1 and w ≥ min_share."""
    units = round((1 - k * min_share) / step)
    rows = [c + (units - sum(c),) for c in itertools.product(range(units + 1), repeat=k - 1) if sum(c) <= units]
    return min_share + step * np.array(rows, float)


def fit_shares(cat: Catalogue, idx: list[int], target: Target, p: Params = P, polish: bool = False):
    """Best shares for a fixed subset: whole grid in one batch, then optional SLSQP polish from the best point."""
    G = simplex_grid(len(idx), p.min_share, p.step)
    H = objective(cat, idx, G, target, p)
    w, h = G[H.argmax()].copy(), float(H.max())  # copy: G is the shared cached grid
    if polish and len(idx) > 1:
        res = minimize(lambda x: -objective(cat, idx, x, target, p)[0], w, method="SLSQP",
                       bounds=[(p.min_share, 1.0)] * len(idx), constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}])
        x = np.clip(res.x, p.min_share, 1.0)
        x /= x.sum()
        hx = float(objective(cat, idx, x, target, p)[0])
        if hx > h:  # SLSQP can stall on the max() kinks; the grid point stays as the fallback
            w, h = x, hx
    return w, h


def solo_profiles(cat: Catalogue, p: Params = P) -> np.ndarray:
    n = len(cat.cheeses)
    return predict(cat, list(range(n)), np.eye(n), p)


def allowed(cat: Catalogue, target: Target, p: Params = P) -> list[int]:
    solo = solo_profiles(cat, p)
    return [
        i for i, c in enumerate(cat.cheeses)
        if (not target.milk or c.milk in target.milk)
        and c.id not in target.exclude
        and not any(solo[i, COL[f]] >= p.exclude_level for f in target.exclude_families if f in COL)
    ]


def describe(cat: Catalogue, idx: list[int], w, target: Target, p: Params = P) -> dict:
    w = np.asarray(w, float)
    order = np.argsort(-w, kind="stable")
    idx, w = [idx[i] for i in order], w[order]
    prof = predict(cat, idx, w, p)
    total, terms, E = score(cat, idx, w, prof, target, p)
    comp = perceived_compounds(cat, idx, w, prof)[0]
    fp = comp @ cat.emb
    t = target_vector(cat, target)
    return {
        "shares": {cat.cheeses[i].id: round(float(x), 3) for i, x in zip(idx, w)},
        "names": [cat.cheeses[i].name for i in idx],
        "score": round(float(total[0]), 4),
        "terms": {k: round(float(v[0]), 4) for k, v in terms.items()},
        "effective_families": round(float(E[0]), 2),
        "profile": dict(zip(DIMS, np.round(prof[0], 2).tolist())),
        "compounds": {COMPOUNDS[k]: round(float(comp[k]), 2) for k in np.argsort(-comp)[:10] if comp[k] > 0.05},
        "character": round(float(character(cat, fp[None], t)[0]), 3) if np.linalg.norm(t) > 0 else None,
        "fingerprint": np.round(fp, 4).tolist(),
        "white_angle": round(float(np.arccos(np.clip(cosine(fp[None], cat.white)[0], -1, 1))), 3),  # radians
    }


def optimize(cat: Catalogue, target: Target, p: Params = P, top: int = 3) -> list[dict]:
    """Candidate retrieval -> beam search over subsets (grid shares) -> SLSQP polish -> diversify by lead cheese."""
    ok = allowed(cat, target, p)
    if not ok:
        return []
    solo = solo_profiles(cat, p)
    found = {(i,): (np.ones(1), float(objective(cat, [i], [[1.0]], target, p)[0])) for i in ok}

    # Candidates: providers for each high-importance dimension (even if weak alone) + best solo cheeses.
    dims = _active(target)
    hi = [d for d in dims if target.dims[d].importance >= 4] or sorted(dims, key=lambda d: -target.dims[d].importance)[:2]
    cands = []
    for d in hi:
        cands += sorted(ok, key=lambda i: solo[i, COL[d]], reverse=target.dims[d].value >= 5)[:5]
    cands += sorted(ok, key=lambda i: -found[(i,)][1])[:20]
    cands = list(dict.fromkeys(cands))[: p.candidates]

    frontier = sorted(found, key=lambda s: -found[s][1])[: p.beam]
    for _ in range(2, p.k_max + 1):
        layer = {}
        for s in frontier:
            for c in cands:
                key = tuple(sorted(s + (c,)))
                if c not in s and key not in found and key not in layer:
                    layer[key] = fit_shares(cat, list(key), target, p)
        found |= layer
        frontier = sorted(layer, key=lambda s: -layer[s][1])[: p.beam]

    picks, leads = [], set()  # best subset per lead cheese, then polish those
    for s in sorted(found, key=lambda s: -found[s][1]):
        lead = s[int(np.argmax(found[s][0]))]
        if lead not in leads:
            leads.add(lead)
            picks.append(s)
        if len(picks) == 2 * top:
            break
    polished = sorted(((s, *fit_shares(cat, list(s), target, p, polish=True)) for s in picks), key=lambda t: -t[2])
    out, leads = [], set()  # polish can shift the lead, so dedupe again on the final shares
    for s, w, _ in polished:
        r = describe(cat, list(s), w, target, p)
        if next(iter(r["shares"])) not in leads:
            leads.add(next(iter(r["shares"])))
            out.append(r)
    return out[:top]


def mix_everything(cat: Catalogue, target: Target, p: Params = P) -> dict:
    ok = allowed(cat, target, p)
    return describe(cat, ok, np.full(len(ok), 1 / len(ok)), target, p)
