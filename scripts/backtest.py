"""Anonymous dish description -> intent -> engine -> top 3, scored against the human blend.
The held-out blend is removed from harmony co-occurrence, and the critic is off so the LLM can't steer to the answer.
Run: python -m scripts.backtest"""

import json

from cheeseblend import agent
from cheeseblend.catalogue import DATA, load


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b)


rows = []
for b in json.loads((DATA / "blends.json").read_text("utf-8")):
    if "split" not in b:
        continue
    cat = load(holdout=b["name"])
    style = {c.id: c.style for c in cat.cheeses}
    human = set(b["cheeses"]) & set(style)
    missing = set(b["cheeses"]) - human
    _, results, _ = agent.recommend(cat, b["description"], rounds=0)
    top = [set(r["shares"]) for r in results]
    exact = max((jaccard(t, human) for t in top), default=0)
    by_style = max((jaccard({style[c] for c in t}, {style[c] for c in human}) for t in top), default=0)
    share_error = next((sum(abs(r["shares"][c] - b["cheeses"][c]) for c in human) / len(human)
                        for r in results if set(r["shares"]) == set(b["cheeses"])), None)
    rows.append({"name": b["name"], "split": b["split"], "human": b["cheeses"], "missing": sorted(missing),
                 "top3": [r["shares"] for r in results], "exact": round(exact, 2), "style": round(by_style, 2),
                 "share_error": share_error and round(share_error, 3)})
    print(f"{b['split']:5} {b['name']:22} exact {exact:.2f} style {by_style:.2f} "
          f"share_err {share_error if share_error is None else round(share_error, 3)} missing {sorted(missing)}")
    for r in results:
        print(f"      {r['score']:.3f} {r['shares']}")

metrics = {}
for split in ("train", "test"):
    rs = [r for r in rows if r["split"] == split]
    metrics[split] = {"n": len(rs), "exact_hit@3": sum(r["exact"] >= 0.5 for r in rs),
                      "style_hit@3": sum(r["style"] >= 0.5 for r in rs)}
print(json.dumps(metrics))
(DATA / "backtest.json").write_text(json.dumps({"metrics": metrics, "rows": rows}, ensure_ascii=False, indent=1), "utf-8")
