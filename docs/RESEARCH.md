# CheeseBlend: research summary, the path we took, and how we use Muse

## The question

Can you predict how a blend of cheeses will taste from published chemistry, and then search for the best blend for a dish? Chefs blend cheeses all the time (fondue moitié-moitié, quattro formaggi, the pizza mozzarella and provolone mix), but by feel. We wanted a model that explains *why* a blend works and can propose new ones.

## Research summary: what the model stands on

| Idea | Source | How we use it |
|---|---|---|
| A compound is smelled in proportion to its concentration over its odor threshold (odor activity value, OAV) | Rothe & Thomas 1963; Grosch 2001. Key cheese odorants by GC-olfactometry: Curioni & Bosset 2002 | 33 key cheese odorants in 13 families; measured concentrations ÷ thresholds in water |
| Perceived intensity grows as a power of stimulus | Stevens 1957; odor exponents 0.2 to 0.7 (Cain 1969) | Intensity = OAV^0.5 |
| Strong odors suppress weaker ones in a mixture | Cain & Drexler 1974; Laing & Willcox 1983 | Masking: F = I·(I/max I)^0.5 (formula is ours) |
| Molecules can be placed in a perceptual space learned from odor labels | Sanchez-Lengeling et al. 2019 (Leffingwell dataset); Lee et al. 2023 (Principal Odor Map) | 3,522 molecules × 113 odor words → PPMI + SVD → 24-d odor-word embedding (Levy & Goldberg 2014) |
| A mixture is the sum of its components' vectors; perceptual similarity tracks the angle between mixtures | Snitz et al. 2013; Ravia et al. 2020 (49,788 human similarity judgments, olfactory metamers) | Blend fingerprint = Σ intensity × compound vector; cosine for "character" match and "smells like" |
| Large mixtures converge on one percept ("olfactory white") | Weiss et al. 2012 | Reproduced in our data: single cheeses 0.33 rad apart, random 32-cheese mixes 0.08 rad. Explains why "mix everything" fails |
| Foods that share compounds pair well (Western cuisines) | Ahn et al. 2011 | Harmony term: cosine of compound fingerprints |
| Recipe co-occurrence as pairing evidence | FlavorGraph, Park et al. 2021 (about 1M recipes) | Harmony term: cheese-to-cheese NPMI |
| Moderate complexity is liked most | Berlyne 1971 (Wundt curve) | Bonus for about 3.5 effective aroma families |
| Each sensory dimension has an ideal level | Moskowitz 1972 (bliss point) | Gaussian fit to each target dimension |

What is ours, and documented as an assumption: the masking formula, the per-family intensity scale, fusing measured and Muse estimates, the melt and stretch interaction rules, every weight, and the ultimate-cheese ideal profile.

## The path we took

The honest version, including what broke.

1. **Plan and heuristics (30 cheeses).** OAV → Stevens → masking → bliss-point fit → beam search. Tests on synthetic cheeses passed. Then real data arrived.
2. **Real data broke the scale.** Thresholds span 6 orders of magnitude (methanethiol 0.02 ppb, acetic acid 22,000 ppb). With one global scale, sulfur and malty compounds drowned everything: Roquefort read blue 1.6/10. **Fix:** a per-family scale anchored to Muse's 0-10 sensory scores by median ratio. Chemistry orders cheeses within a family; the sensory scale anchors it.
3. **Measured and estimated disagree.** Rank correlation between measured intensities and Muse's estimates was about 0.4, because papers use different extraction methods. **Fix:** fuse the two on a log scale instead of letting either one override.
4. **Parsing real papers.** Units came as `ppb in water`, `g kg-1`, `% w/w centre`; a 20-hour curd sample and Raclette's salt read as 27% got in. **Fix:** prefix unit parsing, dry-matter rejection, and a cross-check: a measured composition value must be within 3× of Muse's typical value.
5. **Rights.** Some sources were pirated-PDF mirrors or university proxy URLs. **Fix:** dropped.
6. **Complexity bonus versus taste.** The Wundt term let blends that hit "3.5 aroma families" beat better-tasting ones. The aroma floor was tuned on the train split only.
7. **Research-backed aroma.** We replaced hand-made aroma similarity with the published mixture model: compound embeddings from Leffingwell labels, blends as intensity-weighted sums, cosine similarity. Raw cosine sat at 0.5-0.95 for everything (every cheese shares a cheesy/buttery direction), so character is measured after centering on the "cheese white".
8. **More cheese.** 183 attempted, 171 built (Muse's content filter refused 12 every time). 66 cheeses have measured compounds (483 values), 532 of 684 composition values are measured, and there are 642 cited sources.
9. **An honest regression.** With six times more cheese, exact backtest hits fell (train 2/3 → 1/3). Salty brined cheeses act as "salt boosters". We report it; in the app, the Muse critic catches these.
10. **Interface.** Streamlit widgets couldn't do drag-and-drop, so the main stage is a custom component (plain JS, no build step) wired to the Python engine.
11. **The ultimate cheese.** An engine-only search against an ideal crowd-pleaser profile. A twelve-target ideal let mild melters win on box-ticking, so the ideal is flavour-led: savory and nutty first. Current answer: Red Leicester 75% + Sbrinz 25%.
12. **Real photos.** Muse names each cheese's Wikipedia article; Wikipedia supplies the photo and Commons the license. Only freely licensed photos are kept, linked and credited.

## How we use Muse

Every Muse call goes through `https://api.meta.ai/v1` with the OpenAI SDK, and is cached on disk under a hash of its request, so a rebuild never pays twice. 566 calls built the current data.

| Role | Model | Tier | API shape | Calls |
|---|---|---|---|---|
| Research agent: finds published measurements with sources | `muse-spark-1.3-contributor` | Contributor (public research only) | Responses API + `web_search`, citations from `url_citation` annotations | 172 |
| Extractor: research notes → numbers with exact quotes | `muse-wonder-1.0` | Production | Chat + `json_schema` + `reasoning_effort="high"` | 172 |
| Gap filler: 0-10 score + chemical reason for every unmeasured dimension, typical composition | `muse-spark-1.3` | Production | Chat + `json_schema` | 185 |
| Intent parser: craving → target profile, never names a cheese | `muse-spark-1.3` | Production | Chat + `json_schema` | 11 |
| Critic: culinary sanity check → exclude cheeses → re-optimize (up to 2 rounds) | `muse-spark-1.3` | Production | Chat + `json_schema` | 12 |
| Explainer: why the blend works, from the numbers and sources only | `muse-spark-1.3` | Production | Chat | 9 |
| Photo finder: exact English Wikipedia article per cheese | `muse-spark-1.3` | Production | Chat + `json_schema` (batches of 40) | 5 |

**Trust but verify.** Muse's numbers are never taken on faith. Each measurement must have its number in its own quote, the quote must appear in the research notes, and the URL must be among the search citations. Then units are converted and the value is cross-checked against Muse's own typical value.

**Privacy.** The contributor tier (Meta trains on those prompts) is used only for offline research on public papers. Live cravings typed by users go to the production tier only, and the app refuses to start if a live model is ever set to a contributor tier.

**Live use.** In the app, *Ask Muse* runs intent, then optimization, then the critic loop, then the explainer, capped at 5 asks per session. The three dish buttons are precomputed with the same pipeline, so they're instant and free.
