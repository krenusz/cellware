# Chapter 10 write-up: pre-treatment state predicts which lineages survive a drug (closed 2026-11-14, confirmed narrow)

## Question

ReSisTrace (Dai et al. 2024, GSE223003) labels ovarian cancer cells (Kuramochi) with lentiviral barcodes, lets
them divide once, and splits sister cells. One half is sequenced before treatment (pre). The other half gets a
drug (carboplatin, olaparib, NK cells) or a control, and the survivors are sequenced (post).

The chapter asked two things:
1. **Instruction vs selection (primary, as planned):** does the drug *instruct* resistance in cells, or only
   *select* lineages that were already resistant? This transfers chapters 1-2 to drugs.
2. **Rescue (C10-S1/S2):** within a library, does a lineage's **pre-treatment expression state** predict its
   survival beyond its genetic subclone and capture quality?

## Results

| step | test | data | result | verdict |
|---|---|---|---|---|
| C10-S0 | instruction statistic S, drug vs time-matched control | train only; library placebo | untreated libraries already differ by 0.84 (count-matched retry 0.72; bar 0.5) | **stopped**, data-limited |
| C10-S1 | state beyond CNV subclone + QC, train CV | 10,132 train drug lineages, 853 survived | delta AUC 0.055, SE 0.0066, power 0.83 (0.5 shrink) | GO |
| C10-S2 | same model, frozen, one look | **resistrace:A**, 5,194 lineages, 420 survived | AUC 0.660 -> 0.722, **delta 0.062**, lower 97.5% bound **0.046** | **confirmed** |

Models are logistic regressions on drug-library lineages:
- **M0** = library + CNV-proxy subclone (k = 3, expression-smoothed along the genome) + QC (log UMIs,
  log genes, percent mito, log number of pre cells).
- **M1** = M0 + 5 PCs of 2,000 highly variable genes, averaged over the lineage's pre cells.
- **Survived** = the lineage's pre cells link to a post-treatment family (`prer_r_group`).

## What we can say

- **Expression state before treatment carries survival information** beyond a coarse genetic-subclone
  proxy, library and capture quality.
- **The effect held from train to sealed data** (0.055 -> 0.062). Carboplatin, olaparib and NK libraries are
  pooled with the library as a fixed effect; per-drug effects were not tested.
- **Size:** it adds about 40% to what library + subclone + QC give above chance (AUC above 0.5: 0.160 ->
  0.222). Real but modest: an AUC of 0.72 leaves most survival unpredicted.

## What we cannot say

- **Instruction vs selection.** Never tested. The drug is assigned per library, with 2 libraries per arm (1
  pair once time-matched), so no drug-vs-control quantity is replicated. This is now rule H22.
- **That the state is non-genetic.** The CNV proxy has 3 subclones and is inferred from expression; finer
  genetic structure would show up as "state". Separating them needs variant calls from the reads (~564 GB, not
  downloaded).
- **Survival vs capture.** "Linked to a post family" also requires the descendants to be captured and
  barcode-matched. QC and pre-cell count are controlled, but cell cycle and cell size were not removed.
- **Which genes or programmes carry it, or whether it is drug-specific.** One pooled model, 5 PCs, not
  interpreted.

## Value against the literature

- **Dai 2024** already reports pre-resistant transcriptional signatures in this data set, so the direction is
  published.
- **What this adds:**
  - a quantified increment beyond a subclone proxy + QC;
  - a frozen model confirmed on a sealed lineage partition that it never saw.
- This is a disciplined replication with a sharper control, not a new mechanism (H20 value statement, as
  flagged at C10-S1).

## Exposure (accepted, Bence 2026-11-13)

Pooled per-library counts were seen before sealing: pre-resistant families and cells, and matched post cells
(column-shift bug). They are library totals with no lineage-level state link. They could not have shaped the
model, but they are stated in every C10 report.

## Partitions after chapter 10

| dataset | spent | sealed |
|---|---|---|
| resistrace | A (C10-S2) | B, C |

## Process note

- **Primary question stopped at S0 after 4 amendments + a placebo.** The 2 libraries per arm were visible
  in the 2026-11-12 map before any code (RETRO_2026-11-13.md). Lesson adopted as **H22**: name the
  treatment-assignment unit at S0; fewer than 3 per arm for a between-condition contrast = stop.
- **The rescue worked because it asked a within-library question**, where lineages are the replicates and
  the library is a fixed effect.
- **First oracle call crashed before any statistic was computed** (barcodes repeat across libraries). The
  ledger was not written and A was not spent. The one-line fix was logged and committed before the real look.

## Reopen if

- Variant calls per barcode (the reads) separate genetic from non-genetic state. Partition B is the natural
  test, with the same frozen pipeline plus a genetic term.
- A ReSisTrace-like data set has >= 3 libraries per arm (instruction vs selection).

Files:
- Plans and log: EXPERIMENT_LOG C10-S0 ... C10-S2, RETRO_2026-11-13.md, `lit/C10_top5.md`,
  `lit/gap_2026-11-13.md`.
- Code: `c10s0_partition.py`, `c10s0_sim.py`, `c10s0_placebo.py`, `c10s1_select.py`, `c10s2_confirm.py`.
- Selection and ledger: `selection/C10-S2.json`, `selection/C10-S2_model.npz`, `test_ledger.tsv`.
