"""Reference data from published open datasets -> data/odor_space.json, data/recipe_pairs.json.
Stores derived vectors and scores only, never the raw datasets. Run: python -m scripts.build_reference

Odor space: Leffingwell odor labels for 3,522 molecules (Sanchez-Lengeling et al. 2019, CC BY-NC, via Pyrfume).
  Label co-occurrence -> PPMI -> SVD gives an odor-word embedding (Levy & Goldberg 2014: SVD on PPMI ~ word2vec).
  A compound's vector is the normalized mean of its labels' vectors.
Recipe pairs: FlavorGraph ingredient-ingredient NPMI from ~1M recipes (Park et al. 2021, Apache-2.0)."""

import csv
import io
import json
import urllib.request

import numpy as np

from cheeseblend.catalogue import COMPOUNDS, DATA

LEFFINGWELL = "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/leffingwell/"
FLAVORGRAPH = "https://raw.githubusercontent.com/lamypark/FlavorGraph/master/input/"
DIM = 24

CID = {  # PubChem CIDs (public domain)
    "diacetyl": 650, "acetoin": 179, "delta-decalactone": 12810, "delta-dodecalactone": 12844, "butanoic acid": 264,
    "3-methylbutanoic acid": 10430, "hexanoic acid": 8892, "octanoic acid": 379, "acetic acid": 176,
    "propanoic acid": 1032, "4-methyloctanoic acid": 62089, "4-ethyloctanoic acid": 61840, "3-methylbutanal": 11552,
    "2-methylbutanal": 7284, "2-methylpropanal": 6561, "methanethiol": 878, "dimethyl sulfide": 1068,
    "dimethyl disulfide": 12232, "dimethyl trisulfide": 19310, "methional": 18635, "2-heptanone": 8051,
    "2-nonanone": 13187, "2-undecanone": 8163, "ethyl butanoate": 7762, "ethyl hexanoate": 31265,
    "1-octen-3-ol": 18827, "1-octen-3-one": 61346, "phenylacetaldehyde": 998, "2-phenylethanol": 6054,
    "p-cresol": 2879, "indole": 798, "guaiacol": 460, "ammonia": 222,
}
# Not in Leffingwell: labelled by hand from their published odor descriptions (goaty/mutton; pungent).
MANUAL = {"4-methyloctanoic acid": ["animal", "fatty", "waxy"], "4-ethyloctanoic acid": ["animal", "fatty", "waxy"],
          "ammonia": ["pungent"]}

# Our cheese id -> FlavorGraph ingredient name, where FlavorGraph has the cheese.
FG_NAME = {
    "mozzarella": "mozzarella_cheese", "provolone": "provolone_cheese", "cheddar": "cheddar_cheese",
    "gruyere": "gruyere_cheese", "emmental": "emmenthaler_cheese", "parmigiano": "parmesan_cheese",
    "pecorino_romano": ("pecorino_romano_cheese", "romano_cheese", "pecorino_cheese"), "gorgonzola": "gorgonzola",
    "fontina": ("fontina_cheese", "fontina"), "camembert": "camembert_cheese", "roquefort": "roquefort_cheese",
    "gouda": "gouda_cheese", "comte": "comte_cheese", "raclette": "raclette_cheese", "brie": "brie_cheese",
    "taleggio": "taleggio", "manchego": "manchego_cheese", "feta": ("feta_cheese", "feta"),
    "halloumi": "halloumi_cheese", "paneer": "paneer_cheese", "ricotta": ("ricotta_cheese", "fresh_ricotta"),
    "stilton": "stilton_cheese", "chevre_frais": ("goat_cheese", "fresh_goat_cheese", "chevre_cheese"),
    "monterey_jack": "monterey_jack_cheese", "havarti": "havarti_cheese",
    "mascarpone": ("mascarpone", "mascarpone_cheese"), "burrata": "burrata_cheese",
    "cream_cheese": "cream_cheese", "cottage_cheese": "cottage_cheese", "queso_fresco": "queso_fresco",
    "mozzarella_bufala": "buffalo_mozzarella", "oaxaca": "oaxaca_cheese", "danablu": "danish_blue_cheese",
    "cabrales": "cabrales_cheese", "edam": "edam_cheese", "jarlsberg": "jarlsberg_cheese", "colby": "colby_cheese",
    "bel_paese": "bel_paese_cheese", "asiago": "asiago_cheese", "kasseri": "kasseri_cheese", "cotija": "cotija_cheese",
    "queso_chihuahua": "chihuahua_cheese", "ricotta_salata": "ricotta_salata", "grana_padano": "grana_padano",
    "neufchatel": "neufchatel_cheese", "reblochon": "reblochon_cheese", "munster": "muenster_cheese",
    "limburger": "limburger_cheese", "port_salut": "port_salut_cheese", "beaufort": "beaufort_cheese",
    "double_gloucester": "double_gloucester_cheese", "wensleydale": "wensleydale_cheese", "quark": "quark",
    "american_cheese": "american_cheese", "queso_anejo": "anejo_cheese", "brick": "brick_cheese",
    "caciocavallo": "caciocavallo", "pepper_jack": ("jalapeno_jack_cheese", "monterey_jack_pepper_cheese"),
    "mizithra": "mizithra_cheese", "panela": "panela_cheese", "robiola": "robiola_cheese",
    "red_leicester": "red_leicester_cheese", "scamorza_affumicata": "smoked_mozzarella_cheese",
}


def _csv(url: str) -> list[dict]:
    with urllib.request.urlopen(url, timeout=120) as r:
        return list(csv.DictReader(io.TextIOWrapper(r, encoding="utf-8")))


def odor_space() -> dict:
    rows = _csv(LEFFINGWELL + "behavior.csv")
    labels = [k for k in rows[0] if k != "Stimulus"]
    M = np.array([[r[k] == "1" for k in labels] for r in rows], float)  # molecules x labels
    C = M.T @ M
    np.fill_diagonal(C, 0)
    total, marg = C.sum(), C.sum(1)
    with np.errstate(divide="ignore"):
        ppmi = np.maximum(np.log(C * total / np.outer(marg, marg)), 0)
    U, S, _ = np.linalg.svd(np.nan_to_num(ppmi))
    words = U[:, :DIM] * np.sqrt(S[:DIM])
    words /= np.linalg.norm(words, axis=1, keepdims=True)
    word = dict(zip(labels, words))

    by_cid = {r["Stimulus"]: [k for k in labels if r[k] == "1"] for r in rows}
    compounds = {}
    for c in COMPOUNDS:
        tags, source = by_cid.get(str(CID[c])), "Leffingwell"
        if not tags:
            tags, source = MANUAL[c], "manual"
        v = np.mean([word[t] for t in tags], 0)
        compounds[c] = {"cid": CID[c], "labels": tags, "label_source": source, "vec": (v / np.linalg.norm(v)).tolist()}
    return {"dim": DIM, "descriptors": {k: v.tolist() for k, v in word.items()}, "compounds": compounds,
            "attribution": "Leffingwell odor dataset (Sanchez-Lengeling et al. 2019, CC BY-NC 4.0) via Pyrfume"}


def recipe_pairs() -> dict:
    nodes = {r["node_id"]: r["name"] for r in _csv(FLAVORGRAPH + "nodes_191120.csv")}
    ours = {n: k for k, v in FG_NAME.items() for n in ((v,) if isinstance(v, str) else v)}
    pairs = {}
    for r in _csv(FLAVORGRAPH + "edges_191120.csv"):
        if r["edge_type"] == "ingr-ingr":
            a, b = ours.get(nodes[r["id_1"]]), ours.get(nodes[r["id_2"]])
            if a and b and a != b:
                key = "|".join(sorted((a, b)))
                pairs[key] = max(pairs.get(key, 0), round(float(r["score"]), 4))
    return {"pairs": pairs, "found": sorted({x for k in pairs for x in k.split("|")}),
            "attribution": "FlavorGraph recipe co-occurrence NPMI (Park et al. 2021, Sci Rep, Apache-2.0)"}


if __name__ == "__main__":
    space = odor_space()
    (DATA / "odor_space.json").write_text(json.dumps(space, indent=1), "utf-8")
    word = {k: np.array(v) for k, v in space["descriptors"].items()}
    for probe in ("buttery", "cheesy", "nutty", "smoky"):
        near = sorted(word, key=lambda k: -word[k] @ word[probe])[1:6]
        print(f"{probe:8} ~ {near}")
    pairs = recipe_pairs()
    (DATA / "recipe_pairs.json").write_text(json.dumps(pairs, indent=1), "utf-8")
    print(f"recipe pairs: {len(pairs['pairs'])} among {len(pairs['found'])} of our cheeses")
