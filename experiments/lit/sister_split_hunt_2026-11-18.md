# Sister-split data hunt (2026-11-18)

Bence: "go 1 2" (1 = systematic hunt for sister-split data sets; 2 = metadata check of GSE291368). Why: the project's only novel results (chapters 1-2) came from a generative clone model fitted to a sister-split design (sisters of one clone sent to different wells or conditions) and answering a question the authors left open. Every other chapter tested what one data set was built to test. Analysis: Opus 5.5 (two web-fetch subagents; bioRxiv HTML throttled with HTTP 429, so abstracts came from the bioRxiv and arXiv APIs).

**Searched:** gap map with a new `sister_split` topic and 5 sister-split seeds (Actions run 44316270787, `results_lit/gap_2026-11-18_sister.json`, 5,155 works, 53 tagged; neutral-output term filter on); 6 web searches; GEO records for GSE291368, GSE315827; bioRxiv/arXiv abstracts for 5 preprints. Metadata and abstracts only; nothing downloaded or sealed.

## Gates applied
Design is sister-split or clone-split; data public now; treatment-assignment unit with >= 3 per arm (H22); the authors leave a question open that a clone model can answer (H25 shape); the system serves the goal (stem-cell fate or morphogenesis, H16).

## Candidates
| candidate | design | data | verdict |
|---|---|---|---|
| hESC clonal memory, endoderm (bioRxiv 10.64898/2026.09.02.748863, Morris/Lafzi labs) | **sister-well split**: one barcoded culture split into 4 wells, sisters sampled at hESC, primitive streak (day 2) and definitive endoderm (day 4); 3,814 clones (981 kept for fate analysis); scRNA plus paired snRNA/snATAC multiome | **not public**: GEO and code "at the date of publication" | **best fit; wait.** The authors leave open how the state persists across divisions and put residual sister-well discordance down to stochastic sampling without modelling it. A clone model separating inherited state from sampling is ours to do. It reopens chapters 5 and 7 (tests ready). Weekly accession check. |
| Monoclonal embryoids (GSE291368) | pooled single-clone embryoid bodies, genetic screen, 6 samples | public | **fails:** the processed cell tables list only cell, sample, QC and cell type, with no embryoid ID or guide per cell; the unit and the treatment need re-processing of the raw reads. |
| Monoclonal gastruloid fluctuation test (Cell 2026; GSE291244) | each gastruloid grown from one ESC; lineage recorder | public | **fails (unchanged since 2026-11-07):** no public cell-to-gastruloid table; about 18 gastruloids per sealed partition. |
| SciPhy (GSE315827) | lineage-recorder method set | public, 3 samples | fails: too few units. |
| Human HSPC clonal fate modules (Tomei/Naik, bioRxiv 10.1101/2025.07.28.667115) | barcoded human HSPC clones in a 15-lineage organoid, scRNA plus surface protein | not checked (bioRxiv throttled) | **open:** design (split clones?) and data status unknown; same lab as SIS-skew. Recheck the methods. |
| scRIPT (bioRxiv 10.64898/2026.10.14.759042) | ESC to blood/vessel differentiation, lineage method | not checked | fails the design gate: no sister split. |
| Heritable innate-immune states (bioRxiv 10.64898/2026.06.17.732820) | fluctuation tests in clonal macrophage populations | not checked | out of goal (not stem-cell fate). |
| CloneOT benchmark sets (arXiv 2610.16093) | clones before and after a perturbation | public | excluded domain (standing rule). |

## Result
No public data set passes all gates today. The single best target is the hESC endoderm sister-well set, blocked only by its release. Next checks: Tomei/Naik methods and data; the weekly accession check for the hESC set.
