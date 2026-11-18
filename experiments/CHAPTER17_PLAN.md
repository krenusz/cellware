# Chapter 17 plan: one model of clone behaviour across systems (2026-11-18)

Bence: "we spoke about using all our project knowledge to create a general model of cell behaviour, did we try to fit that assumption??"; "can we join multiple public dataset to create a novel dataset that can point to new directions?"; "1 2 than 3". Novelty: lit/novelty/C17-S0.md (no finding published). Analysis and plan: Opus 5.5.

## Why
`core_model.py` (2026-11-06) was meant to be the project's general model: clone state z, memory rho, instruction tau, selection gamma, split, growth and sampling. It was fitted one system at a time (LARRY E53, SIS E52, gastruloids C3-S1), and one cross-system rule was tested once (C5-S2: memory differs by cell class). From chapter 7 on it was not used. The general-model claim has never been tested: a joint fit, then prediction of a system the model never saw.

## Question
Does one generative clone model with shared rules and per-system scales predict an unseen system's clone statistics better than a per-system baseline? And which factors move cells along the inherited fate axis (data join)?

## Steps (H19: test within 3 steps after S0)
- **S0 (this step; no statistic, no sealed read):**
  1. Inventory: per system, can the core express it (configuration exists / needs a time-split or tree-emission step / cannot), unit, partition status, readout, size.
  2. Exclusions: data sets in excluded domains are not used (standing rule).
  3. Join plan: find a TF perturbation data set with per-cell guide calls and a lineage system it can be projected onto (candidates: the cortical Perturb-seq + lineage set; the embryoid TF screen lacks per-cell guides in its processed files).
  4. Pre-register S1-S3 with H15 pre-flight, H21 readout gate, H22 unit per system and power from power.py.
- **S1:** joint hierarchical fit on train portions only (Actions). Shared hyperparameters for memory per division and per event, instruction and selection scales; per-system nuisance.
- **S2:** leave-one-system-out: freeze on the others, predict one system's clone statistics (sister agreement, clone-size and fate-coupling summaries), scored against a per-system fit and a no-sharing baseline.
- **S3:** sealed look on systems with unspent partitions, using a frozen model and pre-registered predictions; any last partition needs Bence's verbatim OK (H13).

## Systems in hand (from S0 inventory, to be completed)
| system | core configuration | sealed partitions |
|---|---|---|
| LARRY cytokine wells | yes (E53) | held-out shard spent |
| SIS-seq / SIS-skew | yes (E52) | held-out shard spent |
| Raj gastruloid clones | yes (C3-S1) | raj_atac A/B sealed; morph A/B spent |
| CellTag LSK | needs a time-split configuration | celltag_lsk C |
| CellTag reprogramming | needs a time-split configuration | celltag_reprog C |
| cHSPCTrack (human HSPC) | needs a time-split configuration | chspctrack B/C |
| expansion-culture barcoded set | needs a time-split configuration | lsc_larry A/B/C |
| embryo lineage trees | needs tree emission | petracer_embryo B (one embryo)/C |
| LARRY in vivo, DARLIN, CAMELLIA, EPI-Clone, MethylTree | not yet checked | not partitioned (seal before any statistic, H1/H24) |
