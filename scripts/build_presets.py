"""Precompute the demo presets so the app serves them with zero API calls. Run: python -m scripts.build_presets"""

import json

from cheeseblend import agent
from cheeseblend.catalogue import DATA, load
from cheeseblend.engine import mix_everything

PRESETS = {
    "Fondue": "Classic Swiss fondue for dipping bread: melts silky smooth, nutty and fruity, a little funky, gently "
              "stretchy.",
    "Pizza": "A pizza topping: very stretchy, mild and milky, a little tangy, browns in spots.",
    "Ultimate grilled cheese": "Ultimate grilled cheese: nutty, sharp, stretchy, gooey, golden.",
}

cat = load()
out = {}
for name, request in PRESETS.items():
    target, results, log = agent.recommend(cat, request)
    out[name] = {
        "request": request, "target": target.model_dump(), "results": results, "critic": log,
        "explanation": agent.explain_blend(cat, request, target, results[0]) if results else "",
        "mix_everything": mix_everything(cat, target),
    }
    print(name, [r["shares"] for r in results])
(DATA / "presets.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
