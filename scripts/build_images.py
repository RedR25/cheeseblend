"""Real cheese photos -> data/images.json. Muse names each cheese's English Wikipedia article, Wikipedia supplies the
lead photo, Wikimedia Commons the license and author. Stores links and credits only, never the images.
Run: python -m scripts.build_images"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from cheeseblend import agent
from cheeseblend.catalogue import DATA, load_cheeses

UA = {"User-Agent": "CheeseBlend/0.1 (hackathon demo, non-commercial)"}  # Wikimedia asks for a descriptive agent
PROMPT = """For each cheese, give the exact title of its English Wikipedia article: the article about the cheese
itself, not a town, region or person. Spell it exactly as Wikipedia does (e.g. "Gruyère cheese", "Bra (cheese)").
If no English article about the cheese exists, return an empty string."""
FREE = re.compile(r"\bCC\b|CC[ -]?BY|CC0|public domain|\bPD\b", re.I)


def titles(cheeses) -> dict[str, str]:
    out = {}
    for k in range(0, len(cheeses), 40):
        batch = cheeses[k:k + 40]
        item = agent._obj({"id": {"type": "string", "enum": [c.id for c in batch]}, "title": {"type": "string"}})
        res = agent._chat_json("wiki_titles", agent.REASON_MODEL, PROMPT,
                               "\n".join(f"{c.id}: {c.name} ({c.origin})" for c in batch),
                               agent._obj({"items": {"type": "array", "items": item}}))
        out |= {r["id"]: r["title"] for r in res["items"] if r["title"].strip()}
    return out


def _get(url: str, tries: int = 6) -> dict:
    for k in range(tries):  # Wikimedia rate-limits bursts: back off on 429/503
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503) or k == tries - 1:
                raise
            time.sleep(3 * (k + 1))


def _bare(url: str) -> str:
    """Drop the ?utm_... tracking query Wikipedia now appends to image URLs."""
    return urllib.parse.urlunsplit(urllib.parse.urlsplit(url)._replace(query=""))


def photo(title: str) -> dict | None:
    """Wikipedia summary -> lead photo. Kept only if the article is about a cheese and the file is freely licensed
    on Commons (local non-free files are skipped)."""
    s = _get("https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title.replace(" ", "_")))
    about = (s.get("description", "") + " " + s.get("extract", "")).lower()
    thumb, orig = s.get("thumbnail", {}).get("source"), s.get("originalimage", {}).get("source")
    if s.get("type") == "disambiguation" or "cheese" not in about or not thumb or not orig or "/commons/" not in thumb:
        return None
    thumb, orig = _bare(thumb), _bare(orig)
    name = urllib.parse.unquote(orig.rsplit("/", 1)[-1])
    q = _get("https://commons.wikimedia.org/w/api.php?action=query&format=json&prop=imageinfo&iiprop=extmetadata"
             "&titles=" + urllib.parse.quote("File:" + name))
    meta = next(iter(q["query"]["pages"].values())).get("imageinfo", [{}])[0].get("extmetadata", {})
    license_ = meta.get("LicenseShortName", {}).get("value", "")
    if not FREE.search(license_):
        return None
    artist = re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip()
    return {"title": s["title"], "thumb": thumb, "page": s["content_urls"]["desktop"]["page"],
            "file": "https://commons.wikimedia.org/wiki/File:" + urllib.parse.quote(name.replace(" ", "_")),
            "license": license_, "artist": artist[:120] or "unknown"}


def _safe(item):
    cid, title = item
    try:
        return cid, photo(title)
    except Exception as e:  # one missing page must not stop the rest; that cheese keeps its drawn icon
        print(f"{cid:24} skipped: {type(e).__name__}")
        return cid, None


if __name__ == "__main__":
    cheeses, path = load_cheeses(), DATA / "images.json"
    named = titles(cheeses)
    found = json.loads(path.read_text("utf-8")) if path.exists() else {}  # keep earlier finds, retry the rest
    with ThreadPoolExecutor(2) as pool:
        found |= {cid: p for cid, p in pool.map(_safe, [x for x in named.items() if x[0] not in found]) if p}
    path.write_text(json.dumps(found, ensure_ascii=False, indent=1), "utf-8")
    print(f"Muse named {len(named)}/{len(cheeses)} articles; {len(found)} have a freely licensed photo")
