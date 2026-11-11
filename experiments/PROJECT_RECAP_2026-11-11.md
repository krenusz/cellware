# Cellware: full project recap (2026-11-11, project PAUSED)

Branch `modernize`, 603 commits. Working notes, not a paper. Companion files: RETRO_2026-11-11c_EXTENDED.md
(extended retro) and lit/review_2026-11-11.md (literature re-review).

## 1. Goal

- **Long term (Bence, 2026-11-05):** model stem-cell fate behaviour, and eventually how cells sharing local
  rules build a patterned tissue (embryo morphogenesis).
- **Method:** a learned cell rule fitted to clonal (barcoded) data, tested on sealed held-out clones.

## 2. The arc in one paragraph

We started from blood (LARRY) and asked whether cytokines instruct or select fate. The answer: mostly
selection on a fate state the clone already carries, with modest instruction. The same "heritable clone
state" appeared in dendritic-cell sisters (SIS-seq) and in gastruloid clones (Raj lab). Then three chapters
tried to explain how that state behaves in tissue and over time (mixing rescue, memory, cells into space);
all three were limited by data. After H19, the last two chapters asked only what the data could power. Two
confirmed results followed: the clone state is heritable and readable early in three systems, and it
predicts concrete fates (neutrophil vs monocyte, reprogramming success) on unseen clones.

## 3. Chapters

| # | question | data | outcome | sealed confirmation |
|---|---|---|---|---|
| E5-E25 | Can a learned cell rule reproduce blood clonal statistics? | LARRY | partly; cross-well r 0.74 in vitro, 0.46 in vivo | E6/E6b looks |
| 1 | Do G-CSF / M-CSF instruct or select neutrophil vs monocyte fate? | LARRY cytokine (Weinreb 2020) | **mostly selection on a pre-existing state, modest instruction (about 0.5 logit)** | E9/E10, E14, B37 |
| 2 | Does the model generalise? Dendritic-cell sister splits | SIS-seq / SIS-skew (Tian 2021) | **DC fate fixed in the clone before the split (H 0.71)**; Bcor fate effect not identifiable; the RNA trace of the state is weak (r 0.25-0.31 held out) | E51, E54 |
| 3 | Clone states in a self-organising tissue | gastruloid clones (Ayyappan 2025) | **A-P propensity is a stable clone state (0.69 held out)**; mixing rescues elongation; memory is lost at passaging | G1, G6, G7 |
| 4 | Why does mixing rescue elongation? | Raj gastruloids | **closed, data-limited**: rescue not replicated on new subclones | G11, C4-S2a (A+B spent) |
| 5 | What carries clone memory, and how fast does it fade? | Raj ATAC passages, GSE291244 | **parked, data-limited**: chromatin axis real at p1, fade not powered (0.14-0.53) | none (A/B sealed) |
| 6 | Can clone-state cells build a gastruloid in silico? | Raj imaging, chimeras | **closed, exploratory**: a Potts model with posterior pull passes in sample; no powered unseen test | none |
| 7 | Where is the fate state carried (chromatin vs RNA)? | CellTag-multi LSK + reprogramming, cHSPCTrack | **confirmed:** heritable (H 0.75) and readable early from RNA in 3 systems (0.35-0.50 bits/cell) and from chromatin in LSK (0.24); the carrier comparison was never powered | A in 3 datasets |
| 8 | Can early RNA predict a specific fate on unseen clones? | same | **confirmed:** Neu vs Mono rho 0.45; reprogramming success rho 0.59. State vs cytokine in human HSPC untested (power 0.78) | B in LSK and reprog |
| 9 | (choosing) | cHSPCTrack, LSK ATAC | **no test:** erythroid/myeloid split degenerate (7 GM clones); LSK chromatin ready (power 0.84), not run; **paused** | none |

## 4. What we can claim (oracle-confirmed on sealed clones)

1. A clone's fate state exists **before** the decision and is **heritable**, across sister wells:
   - LARRY: day-2 bias, near-full memory over 2 days;
   - SIS-seq: H 0.71;
   - CellTag LSK: H 0.75 on A.
2. It is **readable before differentiation from early RNA**: mouse LSK 0.50, reprogramming 0.35, human HSPC
   0.40 bits/cell (A). It is also readable from early chromatin in LSK (0.24).
3. **Specific outcomes are predictable:**
   - neutrophil vs monocyte share (rho 0.45, B);
   - reprogramming success share (rho 0.59, B).
4. **Cytokines in LSK culture act mainly by selection on that state;** instruction is modest (ch1).
5. **In gastruloids a clone's A-P propensity is a stable clone state** (ch3). Memory is lost at passaging
   events, not with divisions.
6. **Method result:** a pretrained foundation model (Geneformer) reads the clone state no better than PCA.

## 5. What stays open

- **Chromatin vs RNA as carrier, and how much of the state an early snapshot covers.** Needs same-cell clone
  multiome: hESC endoderm (Linneberg-Agerholm 2026), DeepTrack 2026.
- **Instruction vs selection in human HSPC.** Not powerable in cHSPCTrack. chspctrack:B/C sealed.
- **Clone memory dynamics** (ch5) and **mechanism of mixing rescue** (ch4). Both are data-limited.
- **The morphogenesis goal.** The rule-to-tissue step (ch6) has no unseen test yet.

## 6. Data and sealed state

| dataset | spent | sealed (kept) |
|---|---|---|
| celltag_lsk | A, B | **C (last)** |
| celltag_reprog | A, B | **C (last)** |
| chspctrack | A | **B, C** |
| raj_drug / raj_morph_0831 | A / AB | C reserves |
| raj_atac | none | A, B |
| gse291244_founders, raj_wnt | none | A/B/C |
| raj_diffassay | none | rep 2 |

Locked, never read: F8MWB5 subclone set; Triandafillou atlas; revision chimeras.

## 7. Process (how the work was done)

- 20 oracle looks in the ledger (test_ledger.tsv), each pre-registered, each spending one partition.
- **19 H-rules** in program.md, most born from a mistake (see the extended retro).
- **15 retros.** Literature gap map (OpenAlex) at each one.
- Compute: GitHub Actions for public data, the PC for sealed data.

## 8. Ready to resume

- **C9-S2 LSK chromatin -> Neu/Mono** is pre-registered, selected and in the oracle. Power about 0.84 on the
  last partition. Not approved.
- **Watch list:** hESC endoderm multiome, GSE324482, E-MTAB-17351, DeepTrack data.
- **Restart rule:** an H19 chapter choice (data in hand, subset structural counts, readout gate, <= 3 steps).
