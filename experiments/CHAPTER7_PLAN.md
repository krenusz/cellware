# Chapter 7 plan: where is a clone's fate state carried before differentiation? (2026-11-11)

Bence: "go plan A but keep in mind C also" (A = CellTag-multi, Jindal 2024; C = hESC to endoderm sister wells,
Linneberg-Agerholm 2026, data not yet public). Retro: RETRO_2026-11-10b.md.

## Why this question (from our own results)

| what we found | where |
|---|---|
| Fate is mostly already set in the clone before the decision (selection on a pre-existing state; modest instruction) | LARRY #1-#3 |
| The clone fate state is heritable across sister wells (H 0.71) | SIS #13 |
| A gastruloid clone's A-P propensity is a stable clone state | #20 |
| **RNA carries only a weak trace of that state** (held-out r 0.25-0.31; a foundation model no better than PCA) | #17, #19 |
| **Chromatin separates A-P clone groups** (gap 0.38, about 5.6 SE), but 5 clones gave no power for a test | C5-S6 |

The open question is where the state lives. Chapter 7 tests whether **early chromatin carries a clone's later
fate better than early RNA**, with hundreds of clones and sealed hold-outs.

**Serves:** program.md long-term goal (stem-cell fate behaviour), first through blood differentiation. It
connects back to chapters 1-2 (same LSK culture type as LARRY) and forward to C (ES cells, then gastruloids).

## Data A: CellTag-multi (Jindal et al. 2024, Nat Biotech; open access, PMC11180607)

**Design (methods only; results not read):**
- **Blood:**
  - 5,500 mouse LSK cells in broad myeloid medium, tagged with CellTag-multi over 2 days.
  - At day 2.5, half the cells are profiled ("state siblings"), split equally between scRNA-seq and scATAC-seq.
    These are different cells, not multiome.
  - The rest is split into **two replicate wells** and profiled at day 5 ("fate siblings").
- **Reprogramming (second system):** MEF to iEP (Hnf4a-Foxa1). Day 3 state siblings; day 12 and 21 fate
  siblings (reprogrammed / dead-end / transition).

**Files:**
- Count matrices: GEO GSE217119 SuperSeries. LSK RNA GSE216606, d2.5 47 MB + eight d5 libraries; LSK ATAC
  GSE216506, d2.5 48 MB.
- Clone tables and processed objects: github morris-lab/CellTag-multi-2023 and Zenodo 23442375 (CC BY 4.0).

**Downloaded and sha256-locked 2026-11-11** (`data/ext_celltag_multi/SHA256SUMS`): the two clone tables.
**Read: column names and structural counts only.** The fate columns (`cell_type2`, `fate`, `fate_pct`,
`cell_type`, `most_dominant_fate*`, `state/fate`) are unread, and so are the paper's Results and figures.
The authors' ML comparison of RNA vs ATAC fate prediction (their Fig. 2) must **not** be read before the
sealed looks.

**Structure (counts only):**

| system | clones | clones with early RNA sibling and >= 3 fate siblings | with early ATAC sibling | with both |
|---|---|---|---|---|
| LSK (d2.5 -> d5) | 6,822 | 993 | 573 | 160 |
| reprogramming (d3 -> d12/21) | 8,050 | 387 | 99 | - |

- Day-5 RNA cells carry the replicate well in the barcode (`d5r1-` / `d5r2-`), so sister-well agreement is
  measurable as in SIS.
- Day-5 ATAC libraries (`d5_1`..`d5_4`) have no replicate label yet. The mapping must be checked from the GEO
  sample records.

## Sealing (before any fate statistic)

- `partitions.py`: **celltag_lsk** (unit = `clone.id`) and **celltag_reprog** (unit = `clone_id`), salted hash,
  5 buckets. Train 40%, A, B, C 20% each.
- Expected per sealed partition (LSK): about 200 RNA state-fate clones, 115 ATAC, 32 with both.
- One look per partition. The last needs Bence's verbatim OK (H13).
- The fate columns are read only through `train_only` (train) or `test_oracle` (sealed).

## Questions and steps

| step | question | statistic (train first, sealed once) | serves |
|---|---|---|---|
| C7-S1 | this plan + sealing + pre-flight | none | - |
| **C7-S2** | **Is fate a heritable clone state here?** Sister wells r1 vs r2 at day 5 | heritable share H of the fate split across sister wells (SIS method, #13), vs within-well null | replicates #13 in a third system; needed before asking where the state lives |
| **C7-S3** | **Does early chromatin carry the clone's fate better than early RNA?** | held-out prediction of clone fate composition (log-loss / Brier) from d2.5 sibling features: RNA (PCA, HVG residuals) vs ATAC (peak LSI / TF-activity scores); a contrast with a pre-registered sign. Paired subset: clones with both siblings (about 32 per partition: power to be computed). Unpaired: compare per-clone scores across the two groups, adjusting for clone size | the chapter question; tests the C5-S6 carrier hypothesis with hundreds of clones |
| C7-S4 | **Core model:** does the core model's latent clone state (z, fitted from fate across wells) line up with the early chromatin axis better than with RNA? | correlation of fitted z with held-out ATAC / RNA predictions | ties the carrier to the model's state (#17 asked this for RNA) |
| C7-S5 | replication in reprogramming (second system) | C7-S3 statistic on celltag_reprog, RNA only if ATAC is underpowered | generality |

**Stopping rule:** if C7-S2 finds no heritable fate (H not above the null), stop. Without a clone state there is
nothing to localise.

## Pre-flight (H15), to be completed in each step's pre-registration

1. **Unit:** clone. Sister cells in two wells are not independent, so the clone is the unit.
2. **Readout:** fate composition of the fate siblings (the authors' cell-type labels; their 14-type scheme
   grouped as in their methods: Ery/Meg, Baso/Eos/Mast, DCs; progenitor/transition clones handled as
   pre-registered).
3. **Used data first:** none of this data has been used by us. The LARRY / SIS results set the expected effect
   sizes (RNA r about 0.3).
4. **Power:** from train only, per step, H14 >= 0.8. The paired RNA+ATAC subset is small (about 32 per
   partition), so the unpaired design is likely primary.
5. **Exposure:**
   - Methods read (design, clone calling, ML protocol).
   - **Results and figures not read.**
   - The abstract says CellTag-multi "characterizes progenitor lineage priming" (a direction hint, disclosed).
6. **Literature:** CoSpar (Wang 2022), Weinreb 2020, Tian 2021. Gap map seeds to add: Jindal 2024, Biddy 2018.

**Known risks:**
- RNA and ATAC state siblings are different cells, so the comparison is across modalities and cells, not
  within a cell.
- Clone calling uses different thresholds per modality (RNA 2-25 tags, ATAC 1-25), so clone-size and
  detection differences must be adjusted for.
- Two "technical" replicate wells are sisters under the same condition (no perturbation). That is fine for
  heritability, but it is not a conditional design like LARRY.

## Keeping C in view (hESC to endoderm, Linneberg-Agerholm 2026; private until publication)

The C design is the same question with a better structure:
- sister wells across stages (hESC p3, primitive streak, DE);
- paired multiome (RNA + ATAC in the same nucleus);
- about 980 analysed clones.

To make chapter 7 carry over:
- **Code:** every C7 statistic is written against a generic clone table (clone, sibling time, assay, well,
  features, fate), not CellTag-specific columns.
- **Paired analysis:** C7-S3's paired analysis becomes C's primary test, because multiome removes the "different
  cells" caveat.
- **Re-open hook:** C also re-opens chapter 5 (fade of clone chromatin across passages p2 -> p3 -> differentiation),
  so the C5-S6 cross-fitted A-P axis code is reused.
- **Watch:** at every retro, check bioRxiv 10.64898/2026.09.02.748863 for publication, and GEO via E-utilities.
  When released: download, sha-lock, partition by clone before any statistic, then pre-register C7-C1.

## Status 2026-11-11: C7-S3, C7-S5 closed on train; C7-S4 done (Bence)

- **C7-S3 and C7-S5** are closed as exploratory. The fate state is readable early in both carriers, and
  chromatin is not privileged (blood RNA > ATAC; reprogramming equal). See RETRO_2026-11-11. All sealed sets
  are kept.
- **C7-S4** (carrier coverage, train): early RNA carries at least as much clone-fate information as a sister
  well's outcome (coverage 1.36 [0.95, 2.19]); the sister-well ceiling is not a valid ceiling.
- **Data search** (`lit/data_search_c7_2026-11-11.md`): no public dataset has clones + same-cell chromatin +
  early/late fate. cHSPCTrack (human HSPC clones, serial d3-d12, RNA) passed pre-flight, was downloaded and is
  partitioned (`chspctrack`).
