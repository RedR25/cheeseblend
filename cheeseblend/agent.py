"""Every Muse call. Each one is cached on disk under a hash of its request, so a rebuild never pays twice."""

import hashlib
import json
import os
import re
from functools import cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .catalogue import AROMA, COMPOSITION, DIMS, SCORED, Catalogue, COMPOUNDS
from .engine import P, Params, Target, optimize

load_dotenv()

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

RESEARCH_MODEL = "muse-spark-1.3-contributor"  # Meta trains on contributor prompts: public research only
EXTRACT_MODEL = "muse-wonder-1.0"
REASON_MODEL = "muse-spark-1.3"

STYLES = ["fresh", "pasta_filata", "soft_bloomy", "washed_rind", "blue", "semi_hard", "alpine_hard", "hard_grana",
          "brined", "cheddar"]

RESEARCH_PROMPT = """Find published, measured values for {cheese} cheese.
Report moisture %, fat %, salt (NaCl) %, pH, and concentrations of these aroma compounds where measured: {compounds}.
For every number give the unit, the sample description (age, origin) and the source URL.
Prefer open-access peer-reviewed papers and USDA FoodData Central.
Only absolute concentrations count (e.g. µg/kg, mg/kg, ppm, ppb). Skip relative peak areas.
If a value is not found, say "not found". Do not estimate."""

THRESHOLD_PROMPT = """Find odor detection thresholds in water for these compounds: {compounds}.
Use one compilation for all of them if possible (e.g. van Gemert, "Compilations of odour threshold values in air,
water and other media"), otherwise one peer-reviewed table. Give the value, unit and source URL for each.
If a value is not found, say "not found". Do not estimate."""

EXTRACT_PROMPT = """Convert the research notes into JSON measurements.
- Only include numbers that appear in the notes. Never compute or convert a number.
- quote: copy the exact sentence or table row from the notes that contains the number, character for character.
- source_url: the URL the notes give for that number.
- name: use exactly one of: {names}.
- For a range, use the lower bound.
Skip everything else."""

PROFILE_PROMPT = """You are a dairy scientist. For the cheese given, use its milk, process, aging and any measured
values provided. Return its typical make, typical composition, and score every dimension 0-10 for a typical sample,
as perceived in the whole cheese (0 = absent, 10 = the most intense of any cheese).
Give a one-line chemical reason per score (pathway, microbe or process step).
filata: true only for pasta filata (stretched-curd) cheeses.
non_melter: true if it softens but does not flow when heated (acid- or heat-set curd, very high salt or low pH)."""

INTENT_PROMPT = f"""Convert the food request into a target cheese-blend profile.
Use only these dimensions: {", ".join(DIMS)}. Include only the dimensions the request implies (usually 3-8).
value: 0-10 (0 = absent, 10 = the most intense of any cheese). importance: 1-5.
mode: "cooked" if the cheese is heated (melted, baked, grilled), otherwise "cold".
milk: restrict only if the request names a milk, otherwise [].
exclude_families: aroma families the request rules out, otherwise [].
Never name a cheese."""

EXPLAIN_PROMPT = """You are a cheese chemist. Explain why this blend fits the target using only the numbers and
sources given. Say which cheese drives which property and the compounds or process behind it.
Call it a model prediction. Max 120 words. No health or nutrition claims."""

CRITIC_PROMPT = """You check cheese blends predicted by a chemistry model. Does this blend make culinary sense for the
request? Return verdict "ok" or "flag", a one-line reason, and when flagged, the ids of cheeses in the blend to
exclude before re-optimizing."""


@cache
def client() -> OpenAI:
    # Lazy, so tests and the preset-only app import without a key.
    return OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"], max_retries=5, timeout=600)


def _cached(kind: str, request: dict, fn):
    key = hashlib.sha1(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    path = RAW / f"{kind}-{key}.json"
    if path.exists():
        return json.loads(path.read_text("utf-8"))
    out = fn()
    RAW.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    return out


def _obj(props: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


def _chat_json(kind: str, model: str, system: str, user: str, schema: dict, **kw) -> dict:
    req = dict(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": {"name": kind, "schema": schema}},
        **kw,
    )
    return _cached(kind, req, lambda: json.loads(client().chat.completions.create(**req).choices[0].message.content))


def _search(kind: str, prompt: str) -> dict:
    req = dict(
        model=RESEARCH_MODEL,
        reasoning={"effort": "medium"},
        tools=[{"type": "web_search", "search_context_size": "high"}],
        input=prompt,
    )

    def run():
        resp = client().responses.create(**req)
        citations = [
            {"url": a.url, "title": a.title}
            for item in resp.output if item.type == "message"
            for block in item.content if block.type == "output_text"
            for a in (block.annotations or []) if a.type == "url_citation"
        ]
        return {"text": resp.output_text, "citations": citations}

    return _cached(kind, req, run)


# ---------- offline: catalogue build ----------

def research(cheese: str) -> dict:
    return _search("research", RESEARCH_PROMPT.format(cheese=cheese, compounds=", ".join(COMPOUNDS)))


def research_thresholds() -> dict:
    return _search("thresholds", THRESHOLD_PROMPT.format(compounds=", ".join(COMPOUNDS)))


def extract(subject: str, notes: dict, names: list[str]) -> tuple[list[dict], int]:
    """Notes -> (measurements the verifier accepts, number it rejected)."""
    measurement = _obj({
        "name": {"type": "string", "enum": names},
        "value": {"type": "number"},
        "unit": {"type": "string"},
        "source_url": {"type": "string"},
        "quote": {"type": "string"},
    })
    out = _chat_json(
        "extract", EXTRACT_MODEL, EXTRACT_PROMPT.format(names=", ".join(names)),
        f"Subject: {subject}\n\nNotes:\n{notes['text']}",
        _obj({"measurements": {"type": "array", "items": measurement}}),
        reasoning_effort="high",
    )
    kept = [m for m in out["measurements"] if verify(m, notes)]
    return kept, len(out["measurements"]) - len(kept)


def profile(cheese: str, notes: dict, measured: list[dict]) -> dict:
    """Gap filler: make, typical composition and a 0-10 score + reason for every scored dimension."""
    score = _obj({"score": {"type": "number"}, "reason": {"type": "string"}})
    schema = _obj({
        "name": {"type": "string"},
        "milk": {"type": "string", "enum": ["cow", "sheep", "goat", "buffalo", "mixed"]},
        "treatment": {"type": "string", "enum": ["raw", "thermized", "pasteurized"]},
        "style": {"type": "string", "enum": STYLES},
        "origin": {"type": "string"},
        "process": _obj({
            "coagulation": {"type": "string", "enum": ["rennet", "acid", "acid_heat", "mixed"]},
            "cooked": {"type": "boolean"},
            "stretched": {"type": "boolean"},
            "mold": {"type": "string", "enum": ["none", "blue", "white", "washed_rind_bacteria"]},
            "smoked": {"type": "boolean"},
            "aging_months": {"type": "number"},
        }),
        "composition": _obj({k: {"type": "number"} for k in COMPOSITION}),
        "scores": _obj({d: score for d in SCORED}),
        "filata": {"type": "boolean"},
        "non_melter": {"type": "boolean"},
    })
    user = f"Cheese: {cheese}\n\nMeasured values:\n{json.dumps(measured, ensure_ascii=False)}\n\nResearch notes:\n{notes['text']}"
    return _chat_json("profile", REASON_MODEL, PROFILE_PROMPT, user, schema)


_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def _readings(token: str) -> set[float]:
    """A number token read with ',' as a thousands separator and as a decimal comma."""
    out = set()
    for candidate in (token.replace(",", ""), token.replace(".", "").replace(",", ".")):
        try:
            out.add(float(candidate))
        except ValueError:
            pass
    return out


def quote_supports(value: float, quote: str, rel_tol: float = 1e-6) -> bool:
    """The number must appear in its own quote."""
    tol = rel_tol * max(1.0, abs(value))
    return any(abs(v - value) <= tol for token in _NUM.findall(quote) for v in _readings(token))


def _squash(s: str) -> str:
    return " ".join(s.split()).lower()


def verify(m: dict, notes: dict) -> bool:
    """Number in its quote, quote copied from the notes, URL actually cited.
    ponytail: proves the extractor matched the research notes, not that the paper says it; fetch source pages for that."""
    cited = {c["url"] for c in notes["citations"]}
    return (
        quote_supports(m["value"], m["quote"])
        and _squash(m["quote"]) in _squash(notes["text"])
        and (m["source_url"] in cited or m["source_url"] in notes["text"])
    )


# ---------- runtime ----------

def intent(request: str) -> dict:
    dim = _obj({"name": {"type": "string", "enum": DIMS}, "value": {"type": "number"}, "importance": {"type": "integer"}})
    schema = _obj({
        "mode": {"type": "string", "enum": ["cooked", "cold"]},
        "dims": {"type": "array", "items": dim},
        "milk": {"type": "array", "items": {"type": "string", "enum": ["cow", "sheep", "goat", "buffalo"]}},
        "exclude_families": {"type": "array", "items": {"type": "string", "enum": AROMA}},
    })
    out = _chat_json("intent", REASON_MODEL, INTENT_PROMPT, request, schema)
    return {**out, "dims": {d["name"]: {"value": d["value"], "importance": d["importance"]} for d in out["dims"]}}


def critic(request: str, blend: dict) -> dict:
    schema = _obj({
        "verdict": {"type": "string", "enum": ["ok", "flag"]},
        "reason": {"type": "string"},
        "exclude": {"type": "array", "items": {"type": "string", "enum": list(blend["shares"])}},
    })
    return _chat_json("critic", REASON_MODEL, CRITIC_PROMPT, json.dumps({"request": request, "blend": blend}), schema)


def explain(payload: dict) -> str:
    req = dict(model=REASON_MODEL, messages=[
        {"role": "system", "content": EXPLAIN_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ])
    return _cached("explain", req, lambda: client().chat.completions.create(**req).choices[0].message.content)


def recommend(cat: Catalogue, request: str, p: Params = P, rounds: int = 2, target: Target | None = None):
    """Craving -> target -> top blends, with up to `rounds` critic vetoes. Returns (target, results, critic_log)."""
    target = target or Target.model_validate(intent(request))
    results, log = optimize(cat, target, p), []
    for _ in range(rounds):
        if not results:
            break
        verdict = critic(request, results[0])
        log.append(verdict)
        if verdict["verdict"] == "ok" or not verdict["exclude"]:
            break
        target.exclude += verdict["exclude"]
        results = optimize(cat, target, p)
    return target, results, log


def explain_blend(cat: Catalogue, request: str, target: Target, result: dict) -> str:
    index = cat.index
    facts = []
    for cid, share in result["shares"].items():
        c = cat.cheeses[index[cid]]
        facts.append({
            "cheese": c.name, "share": share, "style": c.style, "milk": c.milk,
            "measured": [x.model_dump(include={"name", "value", "unit", "source_url"}) for x in c.compounds][:8],
            "composition": {k: v.value for k, v in c.composition.items()},
            "reasons": {d: s.reason for d, s in c.scores.items() if s.score >= 6},
        })
    prediction = {k: v for k, v in result.items() if k != "fingerprint"}  # raw vector is noise to the explainer
    return explain({"request": request, "target": target.model_dump(), "prediction": prediction, "cheeses": facts})
