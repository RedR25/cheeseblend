"""research -> extract -> verify -> gap fill -> data/catalogue/<id>.json
Run: python -m scripts.build_catalogue p1        (or p2, all, or cheese ids)"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from cheeseblend import agent
from cheeseblend.catalogue import COMPOSITION, COMPOUNDS, DATA, Cheese, to_canonical

P1 = {
    "mozzarella": "Mozzarella (low-moisture)", "provolone": "Provolone", "cheddar": "Cheddar (aged)",
    "gruyere": "Gruyère AOP", "vacherin_fribourgeois": "Vacherin Fribourgeois AOP", "emmental": "Emmental",
    "parmigiano": "Parmigiano Reggiano", "pecorino_romano": "Pecorino Romano", "gorgonzola": "Gorgonzola",
    "fontina": "Fontina", "camembert": "Camembert", "roquefort": "Roquefort", "gouda": "Gouda (aged)",
    "comte": "Comté", "raclette": "Raclette",
}
P2 = {
    "brie": "Brie", "taleggio": "Taleggio", "epoisses": "Époisses", "manchego": "Manchego", "feta": "Feta",
    "halloumi": "Halloumi", "paneer": "Paneer", "ricotta": "Ricotta", "stilton": "Stilton",
    "chevre_frais": "Fresh goat cheese", "monterey_jack": "Monterey Jack", "havarti": "Havarti",
    "scamorza_affumicata": "Smoked Scamorza", "mascarpone": "Mascarpone", "idiazabal": "Idiazabal",
}
P3 = {
    "burrata": "Burrata", "cream_cheese": "Cream cheese", "cottage_cheese": "Cottage cheese",
    "queso_fresco": "Queso fresco", "quark": "Quark", "stracchino": "Stracchino (Crescenza)",
    "mozzarella_bufala": "Mozzarella di Bufala Campana", "caciocavallo": "Caciocavallo", "oaxaca": "Queso Oaxaca",
    "kashkaval": "Kashkaval", "brillat_savarin": "Brillat-Savarin", "chaource": "Chaource", "neufchatel": "Neufchâtel",
    "crottin_de_chavignol": "Crottin de Chavignol", "valencay": "Valençay", "munster": "Munster (Alsace)",
    "limburger": "Limburger", "pont_l_eveque": "Pont-l'Évêque", "livarot": "Livarot", "reblochon": "Reblochon",
    "vacherin_mont_d_or": "Vacherin Mont-d'Or", "maroilles": "Maroilles", "danablu": "Danablu (Danish Blue)",
    "cabrales": "Cabrales", "bleu_d_auvergne": "Bleu d'Auvergne", "fourme_d_ambert": "Fourme d'Ambert",
    "shropshire_blue": "Shropshire Blue", "edam": "Edam", "jarlsberg": "Jarlsberg", "colby": "Colby",
    "port_salut": "Port Salut", "tilsit": "Tilsit", "butterkase": "Butterkäse", "bel_paese": "Bel Paese",
    "asiago": "Asiago", "montasio": "Montasio", "tomme_de_savoie": "Tomme de Savoie", "morbier": "Morbier",
    "saint_nectaire": "Saint-Nectaire", "cantal": "Cantal", "caerphilly": "Caerphilly", "wensleydale": "Wensleydale",
    "lancashire": "Lancashire", "cheshire": "Cheshire", "double_gloucester": "Double Gloucester",
    "red_leicester": "Red Leicester", "beaufort": "Beaufort", "appenzeller": "Appenzeller", "abondance": "Abondance",
    "tete_de_moine": "Tête de Moine", "sbrinz": "Sbrinz", "grana_padano": "Grana Padano",
    "pecorino_sardo": "Pecorino Sardo", "pecorino_toscano": "Pecorino Toscano", "mimolette": "Mimolette",
    "piave": "Piave", "kefalotyri": "Kefalotyri", "kasseri": "Kasseri", "sirene": "Sirene (Bulgarian white brine)",
    "telemea": "Telemea", "ricotta_salata": "Ricotta salata", "mahon": "Mahón", "roncal": "Roncal",
    "zamorano": "Zamorano", "tetilla": "Tetilla", "cotija": "Cotija", "queso_chihuahua": "Queso Chihuahua",
    "ossau_iraty": "Ossau-Iraty", "graviera": "Graviera", "brunost": "Brunost (Norwegian brown whey cheese)",
    "garrotxa": "Garrotxa", "bryndza": "Bryndza", "akkawi": "Akkawi",
}
P4 = {
    "robiola": "Robiola", "squacquerone": "Squacquerone", "coulommiers": "Coulommiers",
    "saint_marcellin": "Saint-Marcellin", "delice_de_bourgogne": "Délice de Bourgogne", "banon": "Banon",
    "selles_sur_cher": "Selles-sur-Cher", "sainte_maure": "Sainte-Maure de Touraine", "langres": "Langres",
    "herve": "Herve", "saint_paulin": "Saint-Paulin", "romadur": "Romadur", "olomoucke_tvaruzky": "Olomoucké tvarůžky",
    "stinking_bishop": "Stinking Bishop", "quartirolo": "Quartirolo Lombardo", "bleu_des_causses": "Bleu des Causses",
    "bleu_de_gex": "Bleu de Gex", "valdeon": "Valdeón", "castelmagno": "Castelmagno", "cashel_blue": "Cashel Blue",
    "maasdam": "Maasdam", "samso": "Samsø", "danbo": "Danbo", "esrom": "Esrom", "vasterbotten": "Västerbottensost",
    "prastost": "Prästost", "leyden": "Leyden (Leidse kaas)", "boerenkaas": "Boerenkaas", "bergkase": "Bergkäse",
    "tomme_des_pyrenees": "Tomme des Pyrénées", "laguiole": "Laguiole", "salers": "Salers",
    "toma_piemontese": "Toma Piemontese", "bra": "Bra", "ragusano": "Ragusano", "fiore_sardo": "Fiore Sardo",
    "canestrato_pugliese": "Canestrato Pugliese", "bitto": "Bitto", "monte_veronese": "Monte Veronese",
    "torta_del_casar": "Torta del Casar", "queso_de_la_serena": "Queso de La Serena",
    "san_simon": "San Simón da Costa", "arzua_ulloa": "Arzúa-Ulloa", "majorero": "Majorero",
    "cornish_yarg": "Cornish Yarg", "sage_derby": "Sage Derby", "lincolnshire_poacher": "Lincolnshire Poacher",
    "coolea": "Coolea", "durrus": "Durrus", "american_cheese": "American cheese (processed)",
    "pepper_jack": "Pepper Jack", "brick": "Brick cheese", "dry_jack": "Dry Jack", "queso_anejo": "Queso añejo",
    "panela": "Queso panela", "requeijao": "Requeijão", "queijo_minas": "Queijo Minas frescal", "oka": "Oka",
    "oscypek": "Oscypek", "branza_de_burduf": "Brânză de burduf", "nabulsi": "Nabulsi", "labneh": "Labneh",
    "manouri": "Manouri", "mizithra": "Mizithra", "kopanisti": "Kopanisti", "metsovone": "Metsovone",
    "beyaz_peynir": "Beyaz peynir", "tulum": "Tulum", "chhurpi": "Chhurpi (hard)", "rushan": "Rushan",
    "domiati": "Domiati", "jben": "Jben", "leipajuusto": "Leipäjuusto", "gamalost": "Gamalost",
    "handkase": "Handkäse", "harzer": "Harzer", "l_etivaz": "L'Etivaz", "mondseer": "Mondseer",
    "dorset_blue_vinney": "Dorset Blue Vinney", "gubbeen": "Gubbeen",
}
ALL = P1 | P2 | P3 | P4


MIRRORS = {"electronicsandbooks.com", "slideserve.com", "docslib.org", "pinkbird.org", "firstlight.cn"}


def _citable(url: str) -> bool:
    """Drop document mirrors (no rights to the copy) and institutional proxy URLs (the public can't open them)."""
    host = urlparse(url).netloc.lower().removeprefix("www.")
    return bool(host) and host not in MIRRORS and not host.startswith("www-")


def _plausible(m: dict, typical: dict) -> bool:
    """A measured composition value must be within 3x (pH: 1 unit) of Muse's typical value. Catches wrong samples
    (20-hour curd) and misread units. ponytail: crude band; per-field ranges if it drops real outliers."""
    t = typical[m["name"]]
    return abs(m["value"] - t) <= 1 if m["name"] == "ph" else t / 3 <= m["value"] <= t * 3


def _canonical(measurements: list[dict]) -> list[dict]:
    """Convert to µg/kg or %; drop unconvertible units, uncitable sources and repeat names (first value wins)."""
    out, seen = [], set()
    for m in measurements:
        v = to_canonical(m["name"], m["value"], m["unit"])
        if v is not None and v > 0 and m["name"] not in seen and _citable(m["source_url"]):
            seen.add(m["name"])
            unit = "" if m["name"] == "ph" else "%" if m["name"] in COMPOSITION else "µg/kg"
            out.append({**m, "value": v, "unit": unit, "confidence": "measured"})
    return out


def build_thresholds():
    path = DATA / "thresholds.json"
    notes = agent.research_thresholds()  # cached, so rebuilding is free
    kept, dropped = agent.extract("odor detection thresholds in water", notes, COMPOUNDS)
    out = {m["name"]: {k: m[k] for k in ("value", "unit", "source_url", "quote")} for m in _canonical(kept)}
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    print(f"thresholds: {len(out)}/{len(COMPOUNDS)} compounds, verifier rejected {dropped}")


def build_cheese(cid: str, name: str) -> str:
    notes = agent.research(name)
    kept, dropped = agent.extract(name, notes, COMPOSITION + COMPOUNDS)
    measured = _canonical(kept)
    prof = agent.profile(name, notes, measured)

    composition = {k: {"value": prof["composition"][k], "unit": "" if k == "ph" else "%"} for k in COMPOSITION}
    composition |= {m["name"]: m for m in measured if m["name"] in COMPOSITION and _plausible(m, prof["composition"])}
    compounds = [m for m in measured if m["name"] not in COMPOSITION]
    sources = [u for u in dict.fromkeys([m["source_url"] for m in measured] + [c["url"] for c in notes["citations"]])
               if _citable(u)]

    cheese = Cheese(
        id=cid, name=prof["name"] or name, milk=prof["milk"], treatment=prof["treatment"], style=prof["style"],
        origin=prof["origin"], process=prof["process"], composition=composition, compounds=compounds,
        scores={d: {"score": min(max(s["score"], 0), 10), "reason": s["reason"]} for d, s in prof["scores"].items()},
        filata=prof["filata"], non_melter=prof["non_melter"], sources=sources,
    )
    (DATA / "catalogue" / f"{cid}.json").write_text(cheese.model_dump_json(indent=1), "utf-8")
    n_comp = sum(m["name"] in COMPOSITION for m in measured)
    return (f"{cid:22} {len(notes['citations']):3} citations | measured: {len(compounds):2} compounds, "
            f"{n_comp} composition | verifier rejected {dropped}")


def _safe(item):
    try:
        return build_cheese(*item)
    except Exception as e:  # one bad cheese must not kill the batch; rerun it alone (cache keeps the rest)
        return f"{item[0]:22} FAILED {type(e).__name__}: {e}"


if __name__ == "__main__":
    args = sys.argv[1:] or ["p1"]
    groups = {"p1": P1, "p2": P2, "p3": P3, "p4": P4, "all": ALL}
    todo = {}
    for a in args:
        todo |= groups.get(a) or {a: ALL[a]}
    (DATA / "catalogue").mkdir(parents=True, exist_ok=True)
    build_thresholds()
    with ThreadPoolExecutor(6) as pool:
        for line in pool.map(_safe, todo.items()):
            print(line, flush=True)
