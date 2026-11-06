# Literature sweep 5, 2026-11-06: division of labour in gastruloids

Asked for by Bence before building the gastruloid model ("division of labour ... a lot of
mystery there"). Context: G1 (#20 clone propensity is a stable state; #21 the gastruloid sets
domain size), G2 (#22 mixing *any* two clones rescues elongation, about +1 logit, opposite
propensities not required), G3 (#23 propensity fades over passages). Searched 2026-11-06.
"Read" = text read in the browser or by a fetch agent; "abstract/search only" = not read in full.

## 1. The source claim: what Ayyappan et al. 2025 show and what they propose

Ayyappan, Triandafillou, Sarma, Raj, bioRxiv 2025.07.12.664536 (PMC12338665, v1 read; v2 exists,
not reachable).
- **Shown:** clones have anterior or posterior propensities; pure clones elongate less often
  (bulk about 75%, clone 4 under 10%, clone 6 about 50%); mixing restores elongation. Their own
  text: "Most clone combinations resulted in elongated gastruloids, regardless of whether clones
  in the chimera were both anterior, both posterior, or ... one ... and the other ... posterior."
  Exception: clone 4 + clone 6 (about 20%). Proportions are regulated: the A-P intersection sits
  at about 35% whatever the starting ratio. Propensity was "not preserved through serial
  passaging"; "in most cases, clones lost their characteristic propensity by passage 4".
- **Proposed, not tested:** "Clones appear to divide labor, making distinct contributions
  necessary for proper morphogenesis." Mechanism candidates: chromatin accessibility at Wnt, RA
  and Nodal motifs; Axin2 1.57-fold higher in anterior clones. Nodal inhibition disrupted
  propensity but kept some sorting; RA disrupted both.
- **Their limits:** modest transcriptional differences (230 of 14,883 genes); small n for
  seqFISH (2-5) and drugs (n = 1-3); "the majority (or even all) of these differences" may have
  no functional consequence.
- **Our reading so far:** the rescue is real and large; its *specificity* to opposite
  propensities is not supported (G2: +1.47 vs +1.01, difference unresolved). "Division of
  labour" in the strict A + P sense is the paper's frame, not its result.

## 2. Monoclonal gastruloids: heritable founder states, polyclonal buffering

Regalado, Qiu, ... Shendure, bioRxiv 2025.05.23.655664 ("Lineage recording in monoclonal
gastruloids reveals heritable modes of early development"; read in the browser). A Cell 2026
paper "A fluctuation test for gastruloid heterogeneity" (S0092-8674(26)01203-1) appears to be its
published form (not verified; Cell page blocked).
- Single-mESC gastruloids are very heterogeneous: compositions cluster into non-differentiating
  (ND), somite-like (MESO), spinal-cord-like (SC-NEURO) and hindbrain-like (HB-NEURO).
- "composition-based group identity did correlate with gastruloids' morphological appearances,
  with MESO and SC-NEURO gastruloids more likely to exhibit elongation." A mesoderm-biased *or* a
  neural-biased gastruloid can elongate; non-differentiating ones do not.
- DNA Typewriter trees: fate bias or restriction appears "within a few cell divisions" of seeding;
  more closely related founders give more similar gastruloids: "fluctuations in the intrinsic
  states of mESCs are heritable, and shape their descendants' fates across many cell divisions."
- Key concept: polyclonal aggregates **buffer** founder fluctuations; monoclonal ones, "free from
  such buffering", show them. Same founder clone gave a balanced, a neural-biased and a
  mesoderm-biased gastruloid.
- **For us:** this is the fluctuation-test logic of chapters 1-2 applied to gastruloids, and it
  offers the simplest explanation of #22: mixing averages out clone extremes. It also matches
  G3: heritable over several divisions, lost over a few passages.

## 3. Proportions vs positions: two separable mechanisms

- **Oriola, Torregrosa-Cortés, Arató, ... bioRxiv 2024.12.16.628776** ("Cell-cell communication
  controls the timing of gastruloid symmetry-breaking"; abstract and results read). Aggregates
  from sorted T+ and T- cells at set fractions: "the timing of symmetry breaking and subsequent
  gastruloid elongation is highly dependent on the initial T+ cell population"; model:
  "pluripotent cells inhibit mesoderm differentiation"; "cell-cell communication dictates
  temporal cell proportions, while differential tissue mechanics governs spatial pole
  formation" (surface-tension differences between T+ and T- aggregates, i.e. sorting).
- **McNamara et al. 2024** (Nat Cell Biol; "Recording morphogen signals reveals origins of
  gastruloid symmetry breaking"; search summary only): a 6-hour window where single-cell Wnt
  activity predicts final A-P position; earlier Nodal/BMP (TGF-beta) heterogeneity shapes the
  Wnt response; differential adhesion sorting proposed.
- arXiv 2509.08929 ("Spontaneous elongation of 3D gastruloids from local cell polarity
  alignment"; search only): elongation from local polarity alignment, not morphogens.
- **For us:** this is exactly the split we found. G1a (where a clone goes) = sorting by a clone's
  mechanical / signalling state; G1b (how big each domain is) = a communication feedback on
  proportions. The model should have both: a sorting term on clone state and a proportion
  feedback (Oriola's pluripotent-inhibits-mesoderm is a concrete candidate).

## 4. Rescue by complementation and its limits

- **Braccioli et al. 2025 Dev Cell** ("Identifying cross-lineage dependencies of cell-type-
  specific regulators in mouse gastruloids"; search summary only): MSGN1-null cells form spinal
  cord only when wild-type cells are present; "inter-tissue communication" is required. A
  defect in one lineage is complemented by another population.
- **Chimeric 3D gastruloids, 2022** (bioRxiv 2022.05.25.493377; search only): Brachyury-
  deficient cells have non-autonomous effects on Wnt3a patterning.
- **Mosaic gastruloids, Nat Cell Biol 2026** ("Mosaic gastruloids reveal a temporal restriction
  for developmental cell competition"; search only): p53-null cells out-compete wild type at a
  developmental stage, independent of size; as few as 2 mutant cells impair 150 wild-type cells.
  Competition exists but is stage-restricted; Ayyappan saw no clone takeover.

## 5. Memory of pluripotent states

- **Shaffer et al. 2020 Cell (MemorySeq, Raj lab):** Luria-Delbrück fluctuation analysis with
  bulk RNA-seq finds genes whose states persist over several divisions (cancer line; not mESC).
- Lineages of ESCs show non-Markovian state transitions (PMC8353490; search only): history
  matters.
- Nanog high/low fluctuations are the classic mESC heterogeneity (search only).
- **For us:** G3's half-life of 1.4 passages (about 4-9 divisions) sits between Regalado's
  "heritable across many divisions" and Ayyappan's "lost by passage 4". No published memory
  timescale for gastruloid propensity exists; G3 may be the first number.

## 6. Theory of division of labour

Evolutionary models (Gavrilets 2010; Ispolatov, Ackermann & Doebeli 2012, arXiv 1111.4597;
Rueffler et al. 2012 PNAS; search only): division of labour can arise among genetically identical
cells through interactions alone, without predisposition. Ispolatov: it "occurs spontaneously at
the regulatory level ... and does not require any genetic predisposition for a particular role".
Developmental relevance is indirect, but it warns against reading clone-level propensity as the
cause of the division: interactions can create the roles.

## What this means for the model and for #22

Three explanations of "any mixing rescues", each with a test on data we have:

| hypothesis | idea | prediction on our data |
|---|---|---|
| **H1 buffering** (Regalado) | elongation needs the aggregate's average state in a window; mixing averages clone extremes | rescue fails only when both clones sit on the same side in the relevant dimension (clone 4 + 6 about 20%); pair elongation predicted by a nonlinear (window) function of the *average* clone state, not the average logit |
| **H2 complementation** (Braccioli) | each clone lacks a capacity (a cell type); the partner supplies it | rescue predicted by which cell types each pure clone fails to make (T+ area, neural vs mesoderm); strict A + P labour is one case |
| **H3 proportion feedback** (Oriola) | elongation timing depends on the starting T+ / pluripotent fraction; pure clones start outside the window | per-clone T+ area (`T_overlap - Sheet1.csv`, downloaded) predicts pure-clone elongation; mixing predicts via the averaged fraction |

H1 and H3 differ in what is averaged (a latent state vs a measured T+ fraction); H2 needs per-
clone cell-type capacity. The window form of H1/H3 is the natural model: elongation probability
peaks at an intermediate composition and falls on both sides.

## Gaps

- Not read in full: McNamara 2024, Braccioli 2025, mosaic gastruloids 2026, Cell 2026 version of
  Regalado, Ayyappan v2. lit_gap.py (OpenAlex) not run: this was a topic sweep, not a retro;
  Ayyappan 2025 and Regalado 2025 should be added as gap-map seeds at the next retro.
- No paper tests division of labour against generic mixing with an effect size; G2 is the first
  such estimate we know of.

## Update after G2b (2026-11-06)

The retest (G2b) keeps "any mixing rescues" but adds a complementarity bonus at the edge of
significance: complementary pairs +0.09 [0.001, 0.17] over same-propensity pairs, and only they
beat their better member. So H1 (buffering) alone is not enough; the model needs a generic term
plus an A-P complementarity term, and H2/H3 must explain both. Clone 4 (severe defect, partly
rescued) is the cleanest single test case.

## Update after G4 (2026-11-06)

Tested on the pair table by held-out prediction: H1 (window on averaged latent state) and H2
(OR/AND complementation) both predict unseen mixtures worse than the additive model; strict
complementation is contradicted (clone 4 pairs below their better member); T+ fraction (H3) does
not explain pure-clone elongation. The rescue is real but its mechanism is not identified by
elongation counts. Needs per-clone cell-type composition (e.g. the Shendure monoclonal data,
GSE291244) or time-resolved T dynamics (Oriola-style) to separate the mechanisms.
