"""CheeseBlend UI: a thin Streamlit layer over the engine. Run: streamlit run app.py"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

from cheeseblend import agent
from cheeseblend.catalogue import AROMA, COMPOUNDS, DATA, DIMS, FAMILY_OF, FUNCTION, TASTE, TEXTURE, load
from cheeseblend.engine import (COL, P, Target, allowed, character, cosine, describe, fingerprint, mix_everything,
                                optimize, perceived_compounds, predict, score, solo_profiles)

# The main stage: drag-and-drop cheese board (cheeseboard/index.html, no build step). Dishes are precomputed with
# Muse (scripts/build_presets.py), the ultimate cheese is the engine alone, and "Ask Muse" is live.
cheeseboard = components.declare_component("cheeseboard", path=str(Path(__file__).parent / "cheeseboard"))

LIVE_CAP = 5  # live Muse asks per session
# Live cravings are user text, so they go to the production tier only. The contributor tier (Meta trains on those
# prompts) is reserved for offline research on public papers.
assert "contributor" not in agent.REASON_MODEL, "live calls must use the production Muse tier"

ULTIMATE = Target(mode="cooked", dims={  # the ideal crowd-pleaser, flavour-led: savory and nutty first, then melt
    "umami": {"value": 8, "importance": 5}, "malty_nutty": {"value": 7, "importance": 5},
    "melt": {"value": 7, "importance": 4}, "buttery": {"value": 5, "importance": 3},
    "ammonia": {"value": 0, "importance": 3}, "fruity": {"value": 4, "importance": 2},
})  # ponytail: a hand-picked ideal; a twelve-target version let mild melters win on box-ticking
INK = RIND = "#2B1A0E"  # cartoon: one ink colour for every outline
COLOR = {  # aroma family -> a cartoon colour you'd guess from its name
    "buttery": "#FFD23F", "cheesy_sweaty": "#FF9F1C", "sour_vinegar": "#A7D129", "goaty": "#C9A27E",
    "malty_nutty": "#A0522D", "sulfur_cooked": "#E0C320", "blue_moldy": "#4D96FF", "fruity": "#FF5D73",
    "mushroom_earthy": "#7D8C5C", "floral_honey": "#FF9ECD", "animal_barnyard": "#8C5A3C", "smoky": "#6B6B6B",
    "ammonia": "#3EC1C9",
}
STYLE_COLOR = {  # cheese style -> cartoon colour of its paste
    "fresh": "#FFFDF2", "pasta_filata": "#FFF3B8", "soft_bloomy": "#FFF6E0", "washed_rind": "#FF9F3F",
    "blue": "#8EB8E8", "semi_hard": "#FFD23F", "alpine_hard": "#FFC21A", "hard_grana": "#F5B23C",
    "brined": "#FFFFFF", "cheddar": "#FF9A1F",
}
MARK = {"color": INK, "line": {"color": "#FFD23F", "width": 3}}
STYLE_NAMES = {
    "fresh": "Fresh & milky", "pasta_filata": "Stretchy", "soft_bloomy": "Soft & bloomy", "washed_rind": "Stinky rind",
    "blue": "Blue", "semi_hard": "Semi-hard", "alpine_hard": "Alpine", "hard_grana": "Hard & grating",
    "brined": "Brined", "cheddar": "Cheddar-style",
}
FRIENDLY = {  # dimension -> (plain-language name, bar colour)
    "buttery": ("Buttery", COLOR["buttery"]), "cheesy_sweaty": ("Cheesy funk", COLOR["cheesy_sweaty"]),
    "sour_vinegar": ("Sharp and sour", COLOR["sour_vinegar"]), "goaty": ("Goaty", COLOR["goaty"]),
    "malty_nutty": ("Nutty", COLOR["malty_nutty"]), "sulfur_cooked": ("Savory onion", COLOR["sulfur_cooked"]),
    "blue_moldy": ("Blue mold", COLOR["blue_moldy"]), "fruity": ("Fruity", COLOR["fruity"]),
    "mushroom_earthy": ("Earthy", COLOR["mushroom_earthy"]), "floral_honey": ("Floral", COLOR["floral_honey"]),
    "animal_barnyard": ("Barnyard", COLOR["animal_barnyard"]), "smoky": ("Smoky", COLOR["smoky"]),
    "ammonia": ("Ammonia", COLOR["ammonia"]),
    "salt": ("Salty", "#7FA8D6"), "sour": ("Tangy", "#C9C230"), "umami": ("Savory", "#C46A3E"),
    "bitter": ("Bitter", "#8C6A4F"), "sweet": ("Sweet", "#E9A23B"),
    "firmness": ("Firm", "#B98F68"), "creaminess": ("Creamy", "#D9B86C"), "crumbliness": ("Crumbly", "#C8943E"),
    "melt": ("Melty", "#FFB627"), "stretch": ("Stretchy", "#FF8A1F"), "browning": ("Browns well", "#B5552B"),
    "oil_off": ("Oily when melted", "#6FA3CC"),
}
DISHES = {"Fondue": ("fondue", "Fondue"), "Pizza": ("pizza", "Pizza"), "Ultimate grilled cheese": ("sandwich", "Grilled cheese")}

pio.templates["cheese"] = go.layout.Template(layout={
    "font": {"family": "Atkinson Hyperlegible, sans-serif", "color": INK, "size": 13},
    "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "#FFF6CC",
    "colorway": ["#FF9F1C", "#4D96FF", "#FF5D73", "#A7D129", "#A0522D", "#3EC1C9", "#E0C320"],
    "polar": {"bgcolor": "#FFF6CC",
              "radialaxis": {"gridcolor": "#F2CF5B", "linecolor": INK, "tickfont": {"size": 9}},
              "angularaxis": {"gridcolor": "#F2CF5B", "linecolor": INK, "linewidth": 3,
                              "tickfont": {"family": "Young Serif, serif"}}},
    "xaxis": {"gridcolor": "#F2CF5B", "zerolinecolor": INK, "linecolor": INK},
    "yaxis": {"gridcolor": "#F2CF5B", "zerolinecolor": INK, "linecolor": INK},
    "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"family": "Young Serif, serif"}},
    "hoverlabel": {"bgcolor": "#FFFBEA", "bordercolor": INK, "font": {"family": "Atkinson Hyperlegible", "color": INK}},
})
pio.templates.default = "plotly_white+cheese"

st.set_page_config(page_title="CheeseBlend", page_icon="🧀", layout="wide")
st.markdown(f"<style>{(Path(__file__).parent / 'style.css').read_text('utf-8')}</style>", unsafe_allow_html=True)
cat = st.cache_resource(load)()


def _read(name):
    path = DATA / name
    return json.loads(path.read_text("utf-8")) if path.exists() else {}


presets, backtest, odor = _read("presets.json"), _read("backtest.json"), _read("odor_space.json")
images = _read("images.json")  # cheese id -> Wikimedia Commons photo + credit (scripts/build_images.py)
labels = {c: v["labels"] for c, v in odor.get("compounds", {}).items()}


@st.cache_resource
def odor_maps(_cat):
    """Solo fingerprints and 2D projections, once per catalogue. Odor axes: PCA of the odor-word vectors."""
    n = len(_cat.cheeses)
    solo = fingerprint(_cat, list(range(n)), np.eye(n), solo_profiles(_cat))
    unit = solo / np.linalg.norm(solo, axis=1, keepdims=True)
    words = np.array(list(_cat.descriptors.values()))
    mu, cmu = words.mean(0), unit.mean(0)
    axes = np.linalg.svd(words - mu, full_matrices=False)[2][:2]
    caxes = np.linalg.svd(unit - cmu, full_matrices=False)[2][:2]
    return (unit, (lambda X: (np.atleast_2d(X) - mu) @ axes.T), (lambda X: (np.atleast_2d(X) - cmu) @ caxes.T),
            (lambda X: np.atleast_2d(X) @ caxes.T))  # last: which way an odor word points in cheese space


solo_fp, word_xy, cheese_xy, word_in_cheese_space = odor_maps(cat)


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def chart(fig, **kw):
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False}, **{"use_container_width": True, **kw})


def holes():
    st.markdown('<div class="cb-holes"></div>', unsafe_allow_html=True)


def radar(blends, target=None):
    """blends: [(label, result)]. Axes: target dims (or all non-aroma dims) + any aroma family present."""
    base = list(target["dims"]) if target else TASTE + TEXTURE + FUNCTION
    dims = list(dict.fromkeys([*base, *(d for d in AROMA if any(r["profile"][d] >= 1 for _, r in blends))]))
    fig = go.Figure()
    for label, r in blends:
        vals = [r["profile"][d] for d in dims]
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=dims + dims[:1], name=label, fill="toself",
                                      opacity=0.5, line={"width": 2.5}))
    if target:
        tgt = [d for d in dims if d in target["dims"]]
        fig.add_trace(go.Scatterpolar(r=[target["dims"][d]["value"] for d in tgt], theta=tgt, mode="markers",
                                      name="target", marker={"size": 12, "symbol": "x", "color": RIND}))
    fig.update_layout(polar={"radialaxis": {"range": [0, 10]}}, height=480, margin={"t": 30, "b": 30})
    return fig


def compound_bars(comp: dict):
    """comp: {compound: perceived intensity}."""
    names = sorted(comp, key=comp.get)
    return go.Figure(go.Bar(
        x=[comp[c] for c in names], y=names, orientation="h", marker_color=[COLOR[FAMILY_OF[c]] for c in names],
        marker_line={"color": RIND, "width": 1.2},
        hovertext=[f"{FAMILY_OF[c]} · smells {', '.join(labels.get(c, []))}" for c in names],
    )).update_layout(height=40 + 24 * len(names), margin={"t": 10, "b": 10}, xaxis_title="perceived intensity")


def _marks(fig, project, marks):
    for name, (v, symbol) in marks.items():
        xy = project(_unit(v))
        fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers+text", text=[name], textposition="top center",
                                 textfont={"family": "Young Serif, serif", "size": 14, "color": INK}, showlegend=False,
                                 name=name, hoverinfo="name", marker={"size": 22, "symbol": symbol, **MARK}))
    return fig


def _zoom(fig, xy: np.ndarray, keep: np.ndarray):
    """Zoom to where the points are (3rd to 97th percentile) so a few outliers don't squash the crowd, always
    keeping the marked points in view."""
    lo = np.minimum(np.percentile(xy, 3, 0), keep.min(0))
    hi = np.maximum(np.percentile(xy, 97, 0), keep.max(0))
    pad = (hi - lo) * 0.08
    return fig.update_xaxes(range=[lo[0] - pad[0], hi[0] + pad[0]]).update_yaxes(range=[lo[1] - pad[1], hi[1] + pad[1]])


WORDS = list(cat.descriptors)
WORD_VECS = np.array(list(cat.descriptors.values()))


def _axes(fig, word_scores: np.ndarray):
    """Name each axis by the odor words at its two ends, so a direction means something ("← fruity … savory →")."""
    def ends(s):
        o = np.argsort(s)
        return ", ".join(WORDS[i] for i in o[:2]), ", ".join(WORDS[i] for i in o[::-1][:2])
    (xl, xr), (yl, yr) = ends(word_scores[:, 0]), ends(word_scores[:, 1])
    ax = {"showgrid": False, "zeroline": False, "showticklabels": False, "showline": False,
          "title": {"font": {"size": 13, "color": INK}}}
    return fig.update_layout(
        xaxis={**ax, "title": {**ax["title"], "text": f"← {xl}                {xr} →"}},
        yaxis={**ax, "title": {**ax["title"], "text": f"← {yl}                {yr} →"}},
        showlegend=False, height=520, margin={"t": 10, "b": 50, "l": 50, "r": 10})


def odor_map(comp: np.ndarray, fp: np.ndarray, marks: dict):
    """Only what matters: the 30 odor words nearest the board, and the molecules actually in it (size = strength)."""
    wxy, cxy = word_xy(WORD_VECS), word_xy(cat.emb)
    near = np.argsort(-(WORD_VECS @ _unit(fp)))[:20]
    fig = go.Figure(go.Scatter(x=wxy[near, 0], y=wxy[near, 1], mode="text", text=[WORDS[i] for i in near],
                               hoverinfo="skip", textfont={"color": "rgba(43,26,14,0.6)", "size": 13}))
    k = [i for i in np.argsort(-comp) if comp[i] > 0.05][:6]
    top = comp[k[0]] if k else 1.0
    fig.add_trace(go.Scatter(
        x=cxy[k, 0], y=cxy[k, 1], mode="markers+text", text=[COMPOUNDS[i] for i in k], textposition="bottom center",
        textfont={"size": 11, "color": INK}, hoverinfo="text",
        hovertext=[f"{COMPOUNDS[i]}: smells {', '.join(labels.get(COMPOUNDS[i], [])) or 'unlabelled'}" for i in k],
        marker={"size": [14 + 26 * comp[i] / top for i in k], "color": [COLOR[FAMILY_OF[COMPOUNDS[i]]] for i in k],
                "opacity": 0.85, "line": {"color": INK, "width": 1.5}}))
    return _axes(_marks(fig, word_xy, marks), wxy)


def cheese_map(fp: np.ndarray, idx: list[int], marks: dict):
    """Cheese neighbourhoods: dots coloured by style (legend on top), the five nearest cheeses named around the star."""
    xy, styles = cheese_xy(solo_fp), [c.style for c in cat.cheeses]
    fig = go.Figure()
    for s in sorted(set(styles), key=lambda s: STYLE_NAMES.get(s, s)):
        k = [i for i, t in enumerate(styles) if t == s]
        fig.add_trace(go.Scatter(
            x=xy[k, 0], y=xy[k, 1], mode="markers", name=STYLE_NAMES.get(s, s), hoverinfo="text",
            hovertext=[f"{cat.cheeses[i].name} · {STYLE_NAMES.get(s, s)}" for i in k],
            marker={"size": 11, "color": STYLE_COLOR[s], "line": {"color": INK, "width": 1.2}}))
    fan = [(0, -70), (95, -38), (90, 42), (-95, 42), (-95, -38)]  # spread the labels around the star, not on top
    for (ax, ay), i in zip(fan, [i for i in np.argsort(-character(cat, solo_fp, fp)) if i not in idx][:5]):
        fig.add_annotation(x=xy[i, 0], y=xy[i, 1], text=cat.cheeses[i].name, ax=ax, ay=ay, arrowhead=0, arrowwidth=1.5,
                           arrowcolor=INK, bgcolor="rgba(255,251,234,0.92)", borderpad=2, font={"size": 12, "color": INK})
    fig = _axes(_marks(fig, cheese_xy, marks), word_in_cheese_space(WORD_VECS))
    fig.update_layout(showlegend=True, height=560, legend={"orientation": "h", "y": 1.13, "x": 0, "font": {"size": 12}})
    return _zoom(fig, xy, cheese_xy(_unit(fp)))


@st.cache_resource
def shelf(_cat):
    """Every cheese for the shelf, with its three loudest flavours as a one-line note."""
    solo, out = solo_profiles(_cat), []
    for i, c in enumerate(_cat.cheeses):
        top = sorted((d for d in AROMA + TASTE if d in FRIENDLY), key=lambda d: -solo[i, DIMS.index(d)])[:3]
        out.append({"id": c.id, "name": c.name, "style": c.style, "milk": c.milk, "origin": c.origin,
                    "notes": ", ".join(FRIENDLY[d][0].lower() for d in top)})
    return sorted(out, key=lambda c: c["name"])


def board_mix(board):
    """[[id, weight]] -> (idx, shares) over cheeses that exist; ([], None) when empty."""
    board = [(i, float(w)) for i, w in board or [] if i in cat.index and float(w) > 0]
    if not board:
        return [], None
    w = np.array([x for _, x in board])
    return [cat.index[i] for i, _ in board], w / w.sum()


def evaluate(board, target):
    """What the board component shows for the current board. target: a Target dict or None."""
    idx, w = board_mix(board)
    if not idx:
        return None
    prof = predict(cat, idx, w[None])
    comp = perceived_compounds(cat, idx, w[None], prof)[0]
    like = character(cat, solo_fp, comp @ cat.emb)
    out = {
        "profile": {d: round(float(v), 2) for d, v in zip(DIMS, prof[0])},
        "compounds": [{"name": COMPOUNDS[k], "family": FAMILY_OF[COMPOUNDS[k]], "smells": labels.get(COMPOUNDS[k], []),
                       "value": round(float(comp[k]), 2)} for k in np.argsort(-comp)[:6] if comp[k] > 0.05],
        "like": [{"id": cat.cheeses[i].id, "name": cat.cheeses[i].name, "style": cat.cheeses[i].style}
                 for i in np.argsort(-like) if i not in idx][:4],
        "confidence": round(float(w @ cat.conf[idx]), 2),
    }
    if target:
        t = Target.model_validate(target)
        out["match"] = describe(cat, idx, w, t)["terms"]["fit"]
        gaps = [(d, float(prof[0, COL[d]]), x.value, x.importance) for d, x in t.dims.items() if d in FRIENDLY]
        gaps = sorted((g for g in gaps if abs(g[1] - g[2]) >= 1.5), key=lambda g: -g[3] * abs(g[1] - g[2]))[:3]
        out["gaps"] = [{"dim": d, "have": round(have, 1), "want": want} for d, have, want, _ in gaps]
        out["fixes"] = best_additions(idx, w, t, out["match"]) if len(idx) < 6 else []
    return out


def best_additions(idx, w, target, base, k=3):
    """Cheeses that raise the craving match most when added at a 25% share: every allowed cheese, tried for real."""
    gains = []
    for c in allowed(cat, target):
        if c not in idx:
            ii, ww = idx + [c], np.r_[w * 0.75, 0.25][None]
            gains.append((float(score(cat, ii, ww, predict(cat, ii, ww), target)[1]["fit"][0]) - base, c))
    return [{"id": cat.cheeses[c].id, "name": cat.cheeses[c].name, "style": cat.cheeses[c].style, "gain": round(g, 3)}
            for g, c in sorted(gains, reverse=True)[:k] if g > 0.005]


def suggestion(run):
    mix = run.get("mix_everything")
    return {"request": run["request"], "target": run["target"]["dims"], "explanation": run.get("explanation", ""),
            "by": run.get("by", "Muse's picks for"), "why": run.get("why", "Why the top pick works"),
            "blends": [{"shares": r["shares"], "names": r["names"], "match": r["terms"]["fit"]} for r in run["results"]],
            "mix": {"n": len(mix["shares"]), "match": mix["terms"]["fit"]} if mix else None}


@st.cache_resource
def ultimate(_cat):
    """The theoretical best cheese: every blend of up to k_max cheeses scored against the ideal profile. Engine only."""
    return {
        "request": "The ultimate cheese", "by": "The engine's picks for", "why": "How it was made",
        "target": ULTIMATE.model_dump(), "results": optimize(_cat, ULTIMATE.model_copy(deep=True)), "critic": [],
        "mix_everything": mix_everything(_cat, ULTIMATE),
        "explanation": f"No Muse call here. The engine tried every blend of up to {P.k_max} of the {len(_cat.cheeses)} "
                       "cheeses against an ideal crowd-pleaser: deeply savory and nutty, melts well, buttery, a touch of "
                       "fruit, and no ammonia. The black ticks on the taste bars are that ideal. The bars show how close "
                       "real cheese can get.",
    }


WEDGE = """<svg class="cb-wedge" viewBox="0 0 150 110" aria-hidden="true">
<path d="M10 58 L122 14 L140 58 Z" fill="#FFE27A" stroke="#2B1A0E" stroke-width="5" stroke-linejoin="round"/>
<path d="M10 58 L140 58 L140 100 L10 100 Z" fill="#FFD23F" stroke="#2B1A0E" stroke-width="5" stroke-linejoin="round"/>
<path d="M126 60 L138 60 L138 98 L126 98 Z" fill="#FF8A1F"/>
<path d="M126 58 L126 100" stroke="#2B1A0E" stroke-width="3"/>
<ellipse cx="66" cy="44" rx="11" ry="4" fill="#D98500"/><ellipse cx="67" cy="45" rx="9.5" ry="3" fill="#F5A815"/>
<ellipse cx="100" cy="32" rx="6" ry="2.4" fill="#D98500"/><ellipse cx="101" cy="33" rx="5" ry="1.8" fill="#F5A815"/>
<circle cx="36" cy="78" r="11" fill="#D98500"/><circle cx="38" cy="80" r="9" fill="#F5A815"/>
<circle cx="74" cy="72" r="6.5" fill="#D98500"/><circle cx="75.5" cy="73.5" r="5" fill="#F5A815"/>
<circle cx="104" cy="84" r="10" fill="#D98500"/><circle cx="106" cy="86" r="8" fill="#F5A815"/>
<circle cx="58" cy="91" r="3.5" fill="#D98500"/><circle cx="59" cy="92" r="2.5" fill="#F5A815"/>
<path d="M22 52 L64 35" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round" opacity="0.85"/></svg>"""


def stamp(n: int) -> str:
    return f"""<svg class="cb-stamp" viewBox="0 0 120 120" aria-label="Every output is a model prediction">
<defs><path id="cb-ring" d="M60,60 m-45,0 a45,45 0 1,1 90,0 a45,45 0 1,1 -90,0"/></defs>
<circle cx="60" cy="60" r="56" fill="#FFE27A" stroke="#2B1A0E" stroke-width="4"/>
<circle cx="60" cy="60" r="35" fill="#FFD23F" stroke="#2B1A0E" stroke-width="2.5" stroke-dasharray="4 3"/>
<text font-family="Atkinson Hyperlegible, sans-serif" font-size="9.5" font-weight="700" fill="#2B1A0E">
<textPath href="#cb-ring" textLength="276" lengthAdjust="spacingAndGlyphs">MODEL PREDICTION · NO HEALTH CLAIMS ·</textPath></text>
<text x="60" y="63" text-anchor="middle" font-family="Young Serif, serif" font-size="25" fill="#FF8A1F"
 stroke="#2B1A0E" stroke-width="1.6" paint-order="stroke">{n}</text>
<text x="60" y="78" text-anchor="middle" font-family="Young Serif, serif" font-size="11" fill="#2B1A0E">cheeses</text>
</svg>"""


st.markdown(f"""<div class="cb-hero">{WEDGE}<div>
<div class="cb-title">Cheese<em>Blend</em></div>
<p class="cb-sub">The right cheese blend for whatever you're making, picked from the flavour chemistry of
{len(cat.cheeses)} cheeses. Built on Meta Muse.</p></div>{stamp(len(cat.cheeses))}</div>""", unsafe_allow_html=True)
if not cat.cheeses:
    st.info("The cave is empty: run `python -m scripts.build_catalogue p1`.")
    st.stop()

by_name = {c.name: c for c in cat.cheeses}
ss = st.session_state
ss.setdefault("live", 0)
ss.setdefault("load_id", 0)


def put_on_board(run):
    """Make `run` the active craving and put its best blend on the board."""
    ss.run, ss.load_id = run, ss.load_id + 1
    ss.board = [[i, max(1, round(s * 100))] for i, s in run["results"][0]["shares"].items()]


if "run" not in ss:  # first visit opens on the first dish, so the board isn't empty
    ss.run, ss.board = None, []
    if presets:
        put_on_board(next(iter(presets.values())))

error, v = None, ss.get("cheeseboard")
if v and v.get("nonce") != ss.get("done_nonce"):
    ss.done_nonce, act = v["nonce"], v.get("action")
    if act == "preset" and v.get("preset") in presets:
        put_on_board(presets[v["preset"]])
    elif act == "ultimate":
        put_on_board(ultimate(cat))
    elif act == "craving":
        if ss.live >= LIVE_CAP:
            error = f"You've used all {LIVE_CAP} Muse asks this session. The dishes still work."
        else:
            ss.live += 1
            try:
                target, results, log = agent.recommend(cat, v["craving"])
                if results:
                    put_on_board({"request": v["craving"], "target": target.model_dump(), "results": results,
                                  "critic": log, "explanation": agent.explain_blend(cat, v["craving"], target, results[0]),
                                  "mix_everything": mix_everything(cat, target)})
                else:
                    error = "No blend fits that craving. Try loosening it a little."
            except Exception as e:
                error = f"Muse couldn't answer that one ({type(e).__name__}). Try again, or pick a dish."
    else:  # "board" or "clear_target"
        ss.board = v.get("board") or []
        if act == "clear_target":
            ss.run = None

cheeseboard(
    cheeses=shelf(cat), paste=STYLE_COLOR, style_names=STYLE_NAMES, family_colors=COLOR, aroma=AROMA,
    dims={d: list(f) for d, f in FRIENDLY.items()},
    presets=[{"name": n, "icon": DISHES.get(n, ("fondue", n))[0], "label": DISHES.get(n, ("fondue", n))[1],
              "request": p["request"]} for n, p in presets.items()],
    suggest=suggestion(ss.run) if ss.run else None, result=evaluate(ss.board, ss.run["target"] if ss.run else None),
    load={"id": ss.load_id, "board": ss.board}, board=ss.board, live_left=LIVE_CAP - ss.live, error=error,
    done_nonce=ss.get("done_nonce", 0), images=images, key="cheeseboard", default=None,
)

holes()
st.markdown("### Go deeper")
tab_sci, tab_cat, tab_bt = st.tabs(["Odor science", "Every cheese", "Accuracy test"])

with tab_sci:
    idx, w = board_mix(ss.board)
    if not idx:
        st.info("Put some cheese on the board above to see its chemistry here.")
    else:
        names = [cat.cheeses[i].name for i in idx]
        st.caption("Your board, as the model sees it. Each compound is a vector in an odor-word embedding learned from "
                   "3,522 molecules' Leffingwell labels; a mix is the intensity-weighted sum of its compounds.")
        prof = predict(cat, idx, w[None])
        comp = perceived_compounds(cat, idx, w[None], prof)[0]
        fp = comp @ cat.emb
        solo = predict(cat, idx, np.eye(len(idx)))
        left, right = st.columns(2)
        with left:
            st.markdown("#### Full profile")
            chart(radar([("your board", {"profile": dict(zip(DIMS, prof[0]))})]
                        + [(n, {"profile": dict(zip(DIMS, s))}) for n, s in zip(names, solo)]))
        with right:
            st.markdown("#### Perceived aroma compounds")
            st.caption("Hover a bar to see how each molecule smells.")
            chart(compound_bars({COMPOUNDS[k]: float(comp[k]) for k in np.argsort(-comp)[:15] if comp[k] > 0.05}))
        marks = {"your board": (fp, "star")}  # one marker: the board cheeses sit almost on top of it anyway
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("#### Where your board sits among smells")
            st.caption("Smells that sit close together smell alike. Grey words are the odor words nearest your board; "
                       "bubbles are its strongest molecules, bigger means stronger.")
            chart(odor_map(comp, fp, marks))
        with m2:
            st.markdown("#### Cheese neighbourhoods")
            st.caption(f"All {len(cat.cheeses)} cheeses, placed by how they smell. Neighbours smell alike. The star is "
                       "your board and the five closest cheeses are named.")
            chart(cheese_map(fp, idx, marks))

    holes()
    st.markdown("#### Find by smell")
    st.caption("Pick odor words. Get the closest cheeses and compounds.")
    words = st.multiselect("Odor words", sorted(cat.descriptors), default=["nutty", "fruity"])
    if words:
        y = np.sum([cat.descriptors[x] for x in words], 0)
        ch = character(cat, solo_fp, y)
        c1, c2 = st.columns(2)
        c1.dataframe([{"cheese": cat.cheeses[i].name, "match": round(float(ch[i]), 3)} for i in np.argsort(-ch)[:8]],
                     use_container_width=True)
        cs = cosine(cat.emb, y)
        c2.dataframe([{"compound": COMPOUNDS[k], "cosine": round(float(cs[k]), 3),
                       "smells": ", ".join(labels.get(COMPOUNDS[k], []))} for k in np.argsort(-cs)[:8]],
                     use_container_width=True)

with tab_cat:
    st.markdown("### The cave")
    st.caption("Every cheese, every number, every source.")
    c = by_name[st.selectbox("Cheese", sorted(by_name))]
    i = cat.index[c.id]
    a, b, d = st.columns(3)
    a.metric("Confidence", f"{cat.conf[i]:.2f}")
    b.metric("Aroma families measured", f"{int(cat.measured[i].sum())}/{len(AROMA)}")
    d.metric("Style", c.style.replace("_", " "))
    st.caption(f"{c.milk} milk · {c.treatment} · {c.origin} · aged {c.process.get('aging_months', '?')} months")
    link = st.column_config.LinkColumn("source")
    st.markdown("#### Composition")
    st.dataframe([{"field": k, "value": v.value, "unit": v.unit, "confidence": v.confidence, "quote": v.quote,
                   "source": v.source_url} for k, v in c.composition.items()], column_config={"source": link})
    st.markdown("#### Measured aroma compounds")
    if c.compounds:
        st.dataframe([{"compound": x.name, "family": FAMILY_OF.get(x.name), "µg/kg": x.value, "quote": x.quote,
                       "source": x.source_url} for x in c.compounds], column_config={"source": link})
    else:
        st.caption("No measured compound data found. Aroma comes from the Muse gap filler and is penalized as uncertain.")
    st.markdown("#### Sensory and function scores")
    st.caption("0-10, from the Muse gap filler, each with its chemical reason.")
    st.dataframe([{"dimension": k, "score": s.score, "reason": s.reason} for k, s in c.scores.items()])

with tab_bt:
    st.markdown("### Proof")
    if backtest:
        st.write("Anonymous dish descriptions (no cheese or dish names) → intent → engine. The held-out blend is "
                 "removed from harmony co-occurrence and recipe data. Five blends is a sanity check, not proof.")
        st.json(backtest["metrics"])
        st.dataframe([{"blend": r["name"], "split": r["split"], "human": r["human"], "top 3": r["top3"],
                       "exact": r["exact"], "style": r["style"], "share error": r["share_error"]}
                      for r in backtest["rows"]])
    else:
        st.info("Run `python -m scripts.backtest` to fill this tab.")

holes()
st.caption("Built during the Meta Global AI Developer Hackathon with Meta Muse models. Odor labels: Leffingwell "
           "dataset (CC BY-NC 4.0, via Pyrfume). Recipe pairs: FlavorGraph (Apache-2.0). Cheese photos: Wikimedia "
           "Commons contributors under their stated licenses; hover a photo for its credit.")
