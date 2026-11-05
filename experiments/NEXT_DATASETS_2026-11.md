# Next conditional data sets: shortlist (2026-11-05)

Goal step: after LARRY's cytokine experiment, fit a second data set where **the same clones are
split across conditions**, then move toward tissue and embryo fate (program.md). Found by web
search on 2026-11-05; data access was read from search summaries and is **not yet verified**
(no files downloaded).

## Criteria

1. Sister cells or clones split across conditions (the conditional design LARRY gave us).
2. Single-cell readout of fate plus a clonal barcode; public counts.
3. A fate question the current model family can pose (instruct vs select, or fate bias).
4. A step toward the long-term goal (other lineages, then multicellular patterning).

## Shortlist (my ranking)

| rank | data set | design | why | access (unverified) |
|---|---|---|---|---|
| 1 | **SIS-seq / SIS-skew**, Tian et al. 2021, Immunity ("Clonal multi-omics reveals Bcor as a negative regulator of emergency dendritic cell development") | Sibling cells of one clone split between conditions: wild-type vs Bcor-deficient siblings grown in parallel, in Flt3L-driven DC development; one sibling profiled, others followed for fate | The closest match to LARRY's design, in a new lineage tree (cDC1, cDC2, pDC). It asks expansion vs fate potential, the same instruct-vs-select question, with a genetic rather than cytokine perturbation | GEO SuperSeries GSE153676 (SubSeries GSE153675); processed data and scripts on Mendeley Data, doi 10.17632/vj9tskkvks.1 |
| 2 | **FateMap**, Goyal et al. 2023, Nature ("Diverse clonal fates emerge upon drug treatment of homogeneous cancer cells") | Barcoded melanoma cells expanded, then split across drug doses and drugs; resistant fate types read by scRNA-seq | A strong conditional design. Resistant fates are largely predetermined but switch with dose and drug: selection vs instruction again, outside stem cells | GEO GSE233622; Goyal lab dataset pages |
| 3 | **Monoclonal gastruloids with DNA Typewriter lineage recording** (GEO GSE291244, 2025) | Gastruloids grown from single mouse ES cells; scRNA-seq plus recorded lineage trees of founder cells; related founders give similar cell-type compositions | The first step toward the long-term embryo-morphogenesis goal: fate bias inherited across sister clades in a self-organising embryo model. Not a perturbation design | GEO GSE291244 |

Also seen, lower priority: ReSisTrace (Dai et al. 2024, Nat Commun; sister-cell priming of drug
resistance, deposition not confirmed); scDynaBar (CRISPR molecular clock, includes gastruloids,
Figshare); ClonMapper (CLL, ENA PRJNA721170).

## Recommendation

Start with **SIS-skew (1)**. It reuses almost everything built for LARRY: the sibling split,
clone-level presence and size statistics, selection vs instruction, held-out discipline. It
changes the lineage tree and the perturbation type. Check first, without downloading the whole
series, that clone-level sibling counts per condition are in the processed Mendeley files. Then
pre-register a data diagnostic (B-series) before any model, as for LARRY.

**Gastruloids (3)** are the natural follow-on for the long-term goal. They link back to the
original 2023-24 multi-agent tissue simulator in this repo.

## Sources

- Tian et al. 2021, WEHI publication page: https://publications.wehi.edu.au/Publications/19066
- SIS-seq data and analysis (Mendeley): https://data.mendeley.com/datasets/vj9tskkvks
- GEO GSE153676 via OmicsDI: https://omicsdi.org/dataset/geo/GSE153676
- FateMap, Goyal et al. 2023: https://www.nature.com/articles/s41586-023-06342-8.pdf ; GSE233622: https://omicsdi.org/dataset/geo/GSE233622
- Monoclonal gastruloids, GSE291244: https://www.omicsdi.org/dataset/geo/GSE291244
- ReSisTrace, Dai et al. 2024: https://www.nature.com/articles/s41467-024-45478-7.pdf
- scDynaBar (Figshare): https://figshare.com/articles/dataset/_b_scDynaBar_b_Using_CRISPR_barcoding_as_a_molecular_clock_to_capture_dynamic_processes_at_single-cell_resolution/27627303

## Check 1: SIS-skew processed data (2026-11-05)

Read without downloading the whole 282 MB Mendeley zip: the zip's file list came from its last
1 MB, and four files were pulled by byte range into `data/sis_seq/fig5/` (uncommitted, like all
raw data): `ST072_Counts.txt`, `ST072_Sorted_cell_number.txt`, `SIS_skew_analysis.rmd` (the
authors' Fig 5 analysis, T. Weber 2020) and a README.

**What the counts are.** 2,607 barcodes x 108 samples, barcode reads per sample (96k-1.1M
reads). Each sample is one well x one sibling arm x one sorted DC type x one PCR replicate:

- Wells W1-W3: wild-type HSPC, Cherry+ = control vector. This is the null: siblings that differ
  only by the label. Wells C1-C3: Cas9 LSK, Cherry+ = Bcor sgRNA (KO siblings), Cherry- =
  untransduced siblings (WT). (Z1-Z3 wells exist; the authors' script ignores them; unknown.)
- Arms: Cherry+ and Cherry-, i.e. **sister cells of one clone under two conditions**.
- Fates: cDC1, cDC2, pDC, sorted; sorted cell numbers given, so reads convert to cells.
- Replicates a/b: the authors keep a barcode only if both replicates see it.

**Per well, after the authors' filters (both replicates, > 1 cell):**

| well | clones | both arms | Cherry+ only | Cherry- only | cells + / - | median size |
|---|---|---|---|---|---|---|
| W1 (null) | 377 | 283 | 50 | 44 | 11.3k / 12.9k | 16 |
| W2 (null) | 295 | 245 | 38 | 12 | 7.5k / 6.1k | 12 |
| W3 (null) | 129 | 121 | 5 | 3 | 0.7k / 0.5k | 3 (authors drop it) |
| C1 (KO) | 436 | 244 | 86 | 106 | 92.7k / 16.3k | 28 |
| C2 (KO) | 420 | 241 | 62 | 117 | 88.5k / 18.1k | 38 |
| C3 (KO) | 436 | 239 | 102 | 95 | 80.2k / 16.8k | 26 |

- KO siblings make about 5x more cells than their WT sisters; null-well arms are balanced. pDC
  presence is higher in KO arms (0.20-0.33 vs 0.08-0.22).
- One-arm clones are far commoner in KO wells (about 45% vs 15-25% in the null): the same
  presence question LARRY raised.
- **Caution:** about a quarter of barcodes recur across wells, so the library is small and a
  barcode is a clone only within one well.

**Verdict: usable.** It is the conditional design we need: clone-level sister counts per
condition, three fates, a built-in null (W wells), about 720 both-arm KO clones and 530 null
clones. The LARRY question becomes: does Bcor loss instruct fate (changes each clone's
cDC1/cDC2/pDC split relative to its sisters) or select (expands clones by their existing bias)?

**Before any model:** pre-register a data diagnostic (B38) with a held-out clone shard sealed
first, exactly as for LARRY: sister concordance of fate fractions in null vs KO wells, KO/WT size
ratio against the clone's WT fate, and one-arm presence by size. Minimum sample: 200 both-arm
clones per arm type.

## Check 3: monoclonal gastruloids, GSE291244 (2026-11-05)

Paper: Shendure lab, "Lineage recording in monoclonal gastruloids reveals heritable modes of
early development" (bioRxiv 10.1101/2025.05.23.655664, PMC12139830; GEO title "A fluctuation
test for gastruloid heterogeneity", PMID 40475526). Code: github.com/shendurelab/DTT (scripts
only, Sections 1-4). Two small annotation files were downloaded to
`data/gastruloid_gse291244/` (uncommitted).

**What GEO holds (processed).**

- sci-RNA-seq3 of pooled monoclonal gastruloids (6 plates, about 144 gastruloids per plate):
  247,064 cells with QC and 16 annotated cell types (hindbrain, somites, spinal cord, NMPs,
  PSC-like, notochord, endoderm, cardiac mesoderm, ...). Matrix 439 MB.
- "tree_of_trees" 10x (106 gastruloids): 66,978 cells, cell type per cell. Matrix 444 MB.
- 10x monoclonal L1-L12 and polyclonal L1-L4 matrices (about 100-190 MB each), mESC multiome
  L1-L4 (about 300 MB each), RAW.tar 560 MB.

**What GEO does not hold:** which gastruloid each cell belongs to, and the DNA Typewriter
lineage. The cell annotations carry no gastruloid id. The authors assign cells to gastruloids
from tape barcodes (DTT `Step3_cell_to_gastruloid.R`), using intermediate files on their lab
server: https://shendure-web.gs.washington.edu/content/members/cxqiu/public/nobackup/sam_tape/
(DebrisSeqTAPE per-gastruloid tape calls, `Tape_cell_uniq.txt`, CellByTape 10x tables,
loci files). Listing seen, nothing downloaded.

**Verdict: usable, with work.** The question it can answer is the long-term one: does fate bias
pass from founder mESC to sister gastruloids (related founders, similar compositions)? Getting
there means re-running the authors' cell-to-gastruloid step from their tape files, then a
gastruloid x cell-type table plus a founder tree. Not a perturbation design. After SIS-skew.

## Check 2: SIS-seq proper, the rest of the Mendeley zip (2026-11-05)

Full zip downloaded (281,734,989 bytes, sha256 813795b4... matches Mendeley) to
`data/sis_seq/full/` and extracted (gitignored; the bundled .git folder was deleted).

**Design (clone_split/).** Each sample is one clone: a few sister cells (2-10 sorted) were
RNA-sequenced as a mini-bulk, while their other sisters were cultured and their DC fate measured
(fractions cDC1 / cDC2 / pDC; the files label cDC1 as CD8+ cDC, cDC2 as CD8- cDC). This pairs
**each clone's state before differentiation with its fate**, the quantity LARRY gave us only at
day 2 (our beta, 7-8 logits per SD).

| batch | clones | with fate | with RNA (data_table.csv) |
|---|---|---|---|
| ILSGR0260r | 40 | 36 | 33 |
| D9JS27 (JS86) | 40 | 28 | 28 |
| JS91 | 40 | 24 | 24 |
| JS96 | 40 | 27 | 0 |

- **85 clones have both** RNA (11,448 genes after the authors' processing; raw, denoised and
  batch-corrected versions all present) and fate. Dominant fate: pDC 40, cDC2 32, cDC1 13; 22 of
  85 are restricted (>= 90% one type).
- **Missing from the zip:** the per-clone sister-vs-sister fate tables (`fate_anno/*/
  *_all_fate_anno.csv`) and the "Revised" annotations; the Rmd reads them from the authors'
  Dropbox. Possibly in GEO GSE153676; not checked yet.
- Also present: Bcor-KO vs WT scRNA-seq (CEL-seq2, 6 plates, gene_count.csv per plate) and the
  CRISPR screen counts.

**What it can answer.** How much of DC fate is already written in a clone's transcriptome before
the split (state predicts fate = substrate for selection), with SIS-skew then asking what Bcor
loss does on top. **Limit:** 85 clones support a state-to-fate map with only a few predictors
(e.g. 2-3 principal components or a pre-chosen gene score), not a genome-wide fit. Minimum
sample for any reading: all 85, with clones split into fit and held-out sets by batch.

## Check 2b: the sister fate tables are in GEO GSE119097 (2026-11-05)

GSE153676 is a SuperSeries: GSE119097 = SIS-seq (158 clones), GSE153675 = Bcor-KO CEL-seq2.
Downloaded to `data/sis_seq/geo/` (gitignored): `GSE119097_family.soft.gz` (30 KB) and
`GSE119097_gene_counts.csv.gz` (1.5 MB, raw counts, 158 clones). Parsed per-clone table:
`sisseq_clones.csv` (parser kept in the session scratchpad, to be moved into experiments/ with B38).

**Per clone, in the sample metadata:** cell counts of cDC1, cDC2 and pDC in **two sister wells**
(sis1, sis2), the filler cells' fate fractions in each well (`*_fill_sis1/2`: the non-clonal
cells in the same well, a per-well background), and how many sisters went to RNA-seq (0-11,
median 4.5).

| batch | clones | both sister wells > 3 cells | one only | median cells per well | RNA |
|---|---|---|---|---|---|
| ILSGR0260r | 40 | 35 | 2 | 111 / 94 | 40 |
| D9JS27 | 39 | 22 | 2 | 7 / 11 | 39 |
| JS91 | 40 | 21 | 4 | 6 / 7 | 40 |
| JS96 | 39 | 27 | 1 | 41 / 46 | 39 |

- **105 clones have both sister wells with > 3 cells, and all 158 have RNA counts** (JS96's
  RNA, absent from the Mendeley tables, is here). So each clone gives: pre-split state, two
  independent sister fate readouts, and a per-well background.
- This is what the instruction-vs-selection logic needs within one condition: sister-sister
  fate agreement beyond the background measures how much fate is fixed in the clone before the
  split; the RNA says what fixes it.
- **No fate statistic was computed** (sister agreement included), so nothing is spent before B38
  is pre-registered with a sealed held-out set.
- Small: 2 of 4 batches have under 12 cells per well, so their fractions are noisy; readings
  must weight by cell count or use the two larger batches.
