"""One live call per API shape we depend on. Run: python -m scripts.smoke"""

import json
import time

from cheeseblend.agent import EXTRACT_MODEL, REASON_MODEL, RESEARCH_MODEL, client


def check(name, fn):
    start = time.time()
    try:
        out = fn()
        print(f"OK   {name} ({time.time() - start:.1f}s): {str(out)[:300]!r}")
    except Exception as e:
        print(f"FAIL {name} ({time.time() - start:.1f}s): {type(e).__name__}: {e}")


schema = {"type": "object", "additionalProperties": False, "required": ["compounds"],
          "properties": {"compounds": {"type": "array", "items": {"type": "string"}}}}

check("chat", lambda: client().chat.completions.create(
    model=REASON_MODEL,
    messages=[{"role": "user", "content": "Name 3 compounds that make Parmigiano taste nutty."}],
).choices[0].message.content)

check("chat+json_schema+reasoning_effort", lambda: json.loads(client().chat.completions.create(
    model=EXTRACT_MODEL,
    reasoning_effort="high",
    messages=[{"role": "user", "content": "List 3 aroma compounds in Gruyère."}],
    response_format={"type": "json_schema", "json_schema": {"name": "Compounds", "schema": schema}},
).choices[0].message.content))


def search():
    resp = client().responses.create(
        model=RESEARCH_MODEL,
        reasoning={"effort": "medium"},
        tools=[{"type": "web_search", "search_context_size": "high"}],
        input="Find one published measurement of free glutamate in Parmigiano Reggiano, with the source URL.",
    )
    cites = [a.url for item in resp.output if item.type == "message" for b in item.content
             if b.type == "output_text" for a in (b.annotations or []) if a.type == "url_citation"]
    return {"text": resp.output_text, "citations": cites}


check("responses+web_search", search)
