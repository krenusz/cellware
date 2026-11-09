# Reading notes, chapter 6 mechanism (C6-S6, 2026-11-09)

Bence: "go read 2" (retro 2026-11-09, failure 2: mechanism literature read after building). Read before any new
code. Paraphrased; no sealed data involved.

## 1. de Jong et al. 2024, PLOS Comput Biol e1011825 (gastruloid shapes; Cellular Potts)

Read: full text except Table 1 and equations (images); **code read as text** (`data/ext_plos_gastruloid/repo`,
commit 04a7a20, Tissue Simulation Toolkit C++, never compiled or run).

- **Data:** E14 gastruloids, 200 cells/well, CHIR 48-72 h, fixed 96 h (n 413, 8 replicates); shape by LOCO-EFA
  modes L2/L1, L3/L1 (scalar aspect ratios judged insufficient). Light-sheet mosaic 91-96 h: cells near the poles
  move most along the long axis; proliferation everywhere (Ki-67). ROCK inhibitor (Y-27632, 72-96 h): less
  elongated, often "snowman" shapes with segregated cell types.
- **Model (from `ce.par`, `J.dat`, `ca.cpp`):** CPM, T 25, target area 100 px, target length 10, lambda 5 / 5,
  200 cells on 500x500, 100,000 MCS. J: cell-medium 10, like-like 10, **unlike 30** (strong type segregation).
  **Initial types random** (`SetRandomTypes`, frac 0.5); cells sort during the run.
- **Crawling (Belmonte-style filopodia):** every 20 MCS each cell links to up to 3 neighbours inside a double cone
  of +-45 degrees around its polarity (distance < r_max); links pull (energy -lambda_F x change in link length,
  lambda_F 15-20). Polarity is updated as 0.99 x own + 0.01 x mean of linked neighbours' polarity (Vicsek-like),
  starting random; the axis emerges. Best variant `pulling_method 1`: **links only between cells of the same
  type**.
- **Fits:** differential adhesion alone fits badly; one-type crawling D 0.40 (p 9e-9); **two types + same-type
  crawling D 0.15 (p 0.15, not rejected)**; yellow-yellow-only crawling also fits (D 0.15, p 0.12).
- **Limits (theirs):** 2-D, no division or differentiation, abstract parameters; symmetry breaking (which pole is
  posterior) is not modelled; the two types are abstract, not fates.
- **For us:** gives a fitted, coded elongation mechanism that starts from mixed types. It does not say where
  each type ends up along the A-P axis, so **it does not answer our clone-placement question by itself.**
  Unresolved from text reading: units of r_max (2) vs cell size; does a run end with one domain per type or
  several (check before relying on it).

## 2. Guan ... Hormoz 2025, bioRxiv 2025.09.27.678924 (agent-based symmetry breaking)

Read: PMC full text; the model equations sit in a separate "Biophysical model.pdf" (not read; kernel and
parameter table therefore unconfirmed).

- Point cells, overdamped, Gaussian noise; 1,500 cells (25% "outer"); short-range linear contact force by type
  pair; optional long-range ("gravitation-like") pairwise attraction by type.
- **Adhesion alone: best asymmetry 0.83, no robust single axis.** Adding long-range **outer-outer attraction**:
  asymmetry up to 2.05, robust region about 18% of adhesion space; it was in 99.9% of the top 1%. Best adhesion:
  strong like-like for the majority type, weak between types.
- Initial condition is ambiguous (random vs radial prepattern); random-mixed starts are not reported. A gene
  network makes the two types emerge from a uniform start (one example).
- Code: promised, not posted.
- **For us:** names the ingredient our particle model lacked: patches (C6-S2) are the expected failure of
  short-range adhesion alone; long-range attraction within the posterior type is the published fix.

## 3. McNamara et al. 2024, Nat Cell Biol 26:1832 (Wnt recorder)

Read: bioRxiv v1 in full; published author manuscript (PMC11806519) partly. Unverified: the published cadherin
gain-of-function result and the code repo (summary only).

- Wnt is uniform during CHIR; a Wnt-inactive population appears at ~90-96 h; **patches without axial bias at 96 h;
  one pole by 108 h.** Live imaging: Wnt-active cells move, cluster and merge.
- **Re-aggregation of cells recorded at 90-96 h: they sort by Wnt state, 0/34 aggregates multipolar.** Sorting of
  a mixed population gives one pole reliably in real cells.
- Wnt-high cells express E-cadherin/EpCAM; Wnt-low cells Pcdh8/Pcdh19 (N-cadherin uniform).
- **Early recordings predict position:** Nodal at 45-48 h marks future anterior, BMP posterior; Wnt at 96 h
  excludes cells from the anterior. Activin pretreatment shrinks the posterior domain (dose-dependent).
- Data GEO GSE274389 (per summary; unverified).
- **For us:** the pole is the sorted Wnt-high cells; position is set by **cell state before sorting**, signalled
  as early as 45-48 h. That is the clone-propensity picture (a clone's state biases its cells' early signalling
  state, sorting does the rest), and the opposite of our C6-S3 to S5 design (field first, cells find it).

## Synthesis: why C6-S2..S5 failed, and the redesign

| our step | design | literature verdict |
|---|---|---|
| C6-S2 | continuous-s adhesion, no long-range term | Guan: adhesion alone gives multi-cluster -> our patches |
| C6-S3 B | Potts, continuous-s J (J0 2, Jd 4-8, medium 8-10), no crawling | de Jong: needs a binary strong unlike/like ratio (30/10) and same-type crawling |
| C6-S3..S5 A | Wnt field forms first, clones find it | McNamara: state first, sorting makes the pole |

**Redesign (to pre-register as C6-S7, exploratory under H17 until a powered test exists):**

1. **Identity from state (existing rule 2):** each cell posterior (Wnt-high type) with the core's proportion
   feedback on s = clone state copy + noise. No Wnt field.
2. **Sorting:** binary type adhesion, strong like-like, weak unlike (de Jong ratio 3:1; Guan), plus **long-range
   attraction among posterior cells** (Guan) so patches coalesce into one pole.
3. **Elongation:** same-type crawling with Vicsek polarity alignment (de Jong best variant), in the posterior
   type (yellow-yellow also fits); clone competence scales crawling strength (handle for T-d, 6a round; cf.
   cadherin switch, Cell Rep 2025).
4. **Readouts:** fraction line scan (as C6-S4), multipolar rate (benchmark: McNamara 0/34), moment elongation.

**Expected:** clone placement follows each clone's posterior share (as in the 1-D core, which fits T-a), now with
the axis built by the cells. The fit targets stay the 5 open clones (exposed, out-of-clone CV only).
**H17:** the S2 power must be computed from this model's predicted 6a+bulk effect before any claim; S1 is
unpowerable under the core (<= 0.20) whatever the mechanism. Unless S2 reaches 0.8, C6-S7 is labelled
exploratory.

**Implementation choice:** port the three rules into our particle model (fast, ours), not run the C++ toolkit
(untrusted code, Qt build, 100,000 MCS per run). The CPM is kept as the cross-check if the particle port fails
step 0.
