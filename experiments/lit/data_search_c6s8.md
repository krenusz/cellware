# C6-S8: public data search for an unseen test of the clone-state model (2026-11-10)

Bence: "1" (search public data while C6-S8a runs). Reading only: no download, no outreach.

**The requirement (C6-S1, C6-S8):** gastruloids with clone- or population-labelled cells, ideally pure and mixed
on the same plate, with outlines or line scans, so the model can predict per-gastruloid placement or elongation
without a new parameter.

| source | what it has | clone or mix labels | n | usable as an unseen test? |
|---|---|---|---|---|
| **Ayyappan 2025, Raj lab** (bioRxiv 2025.07.12.664536; our data store) | clone mixes, line scans, morphology | yes | hundreds | **already ours**: open, spent or sealed (S1, S2 partition C, locked revision sets) |
| **Triandafillou 2026, Raj lab** (eLife RP109268; BioImage Archive S-BIAD3844; Zenodo 10.5281/zenodo.22233771) | seqFISH atlas of 26 gastruloids; brightfield morphology of 529 gastruloids (6 plates, 3 days); per-gastruloid elongation and mixing index computed | **no** (E14TG2a only) | 529 / 26 | **not for the clone question.** It could check non-clone predictions only (spread of elongation across gastruloids, cell-type position vs elongation). Whether per-gastruloid tables are deposited is unverified |
| **Hashmi 2022** (eLife 59371; Zenodo 5727050, 34 GB) | live imaging, optic flow, **surface-tension measurements**, stainings | **no** (wild type) | not stated | not a test. A possible **calibration** source for a Marangoni-type term (surface tensions) |
| **Fiuza 2024** (bioRxiv 2024.12.12.628151) | merged WT + Tbxt-mutant gastruloids | yes (genotype, not clone) | 20 | counts only: 13/20 mix and fail to elongate, 7/20 stay segregated and elongate (seen in a search snippet, so now **exposed**). Too small and qualitative |
| Chimeric 3D gastruloids 2022 (Development dev.200812; Bra-/- + WT) | imaging of ratio mixes | yes (genotype) | ? | checked in C4-S1: no public per-population data found |
| Regalado/Qiu 2026 Cell (monoclonal gastruloids, GSE291244) | founder lineage, monoclonal | monoclonal | many | **parked** (FOCUS: no work without Bence's OK); sequencing, not spatial shape |
| Blotenburg 2025 PLoS One; Fiuza 2024 size series; Trani-Bustos 2025 | aspect ratios per condition | no | tens per condition | shape benchmarks only (used in C6-S7 as a sanity check) |

## Verdict

- **No public dataset outside the Raj store has clone- or population-labelled gastruloids with per-gastruloid
  spatial or shape data.**
- The only unseen clone tests remain the Raj sealed sets: S1 (1+6 pairs, passage 4) and S2 (partition C). Their
  power is being computed in C6-S8a.
- The Raj lab's own 2026 atlas (Triandafillou) is the largest open morphology set (529 gastruloids). It is single
  line, so it can only check the model's non-clone behaviour.
- **What would make chapter 6 testable:** a new experiment, or a future release of same-plate pure + mixed clone
  gastruloids with outlines. The search found no such release. Re-run this search at the next retro.
