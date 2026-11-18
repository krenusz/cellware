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

## S0 inventory and searches (2026-11-18, metadata only; no statistic, no sealed read)
**Pre-protocol data sets (H5 provenance):** `larry_invivo`, `epiclone`, `ext_methyltree`, `darlin`, `camellia` were opened in tracks of 2026-11-02/03, before the sealing rules. They count as **used data**: usable in the S1 joint fit, not as clean held-out targets.

**Join, perturbation side (finding 3):** GSE284197 (Ding/Pollen, Nature 2026; public 2025-10-13): CRISPRi Perturb-seq of 44 TFs in primary human cortical radial-glia cultures (days 0 and 7) plus a 5-TF lineage-tracing and slice-culture set; code at github.com/jding5066/perturbTF. Per-cell guide calls are expected (Perturb-seq); check in S0b. This is the only perturbation set found with per-cell guides in a stem/progenitor system. The embryoid TF screen (GSE291368) lacks per-cell guides in its processed tables.

**Clone-split transplants (Bence's 3):**
| candidate | design | data | verdict |
|---|---|---|---|
| Rodriguez-Fraticelli 2020 (GSE134242, Nature 2020, TCF15) | LARRY-barcoded HSCs into primary recipients (at least 4) and secondary transplants; inDrop HSC and KIT libraries; 90 samples | public | **best transplant target**: if clones are shared across recipients, recipient-split sisters test intrinsic vs extrinsic fate. Recipients are the units (H22 >= 3 per arm needs checking). Also re-analysed by the 2026 kinetics model paper. Check clone sharing across recipients from structure only (S0b). |
| Kinetics-based haematopoiesis model (Nature Cell Biology 2026; Zenodo 14497706, CC-BY, 3 rds files) | single-HSC-derived systems plus polyclonal controls; re-analyses GSE134242 and GSE299000 | public | source of a published "extrinsic regulation" claim: read at claims level before any test (H25). |
| GSE299000 (neonatal platelet-biased HSC fate mapping, 2026) | in situ fate mapping across development, 8 samples | public | not a split design; possible extra system for the S1 fit. |
| Excluded | two disease-domain transplant studies found | | excluded domain (standing rule). |

**Next (S0b):** structure-only checks: GSE284197 per-cell guide columns; GSE134242 clone sharing across recipients (counts only). Then the S1-S3 pre-registration.
