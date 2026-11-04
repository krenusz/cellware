# S series: side experiments on more data and on image-based fate prediction

Started 2026-11-04 at Bence's request: "look for more datasets like what we have, and images too,
so we can maybe fit a DINO-like action predictor". Same rules as the main line: pre-register,
one look per test, public data only, **no outreach** (data "available on request" is out of
scope).

## What an "action predictor" means here

DINO-WM (Zhou et al. 2024, dino-wm.github.io) and DINO-world (Meta FAIR 2025, arXiv 2507.19468)
keep a frozen DINOv2 encoder for observations and train a small predictor that maps (current
embedding, action) to the next embedding. In our biology the natural mapping is:

| world-model term | cellware analogue |
|---|---|
| observation | a cell's brightfield image patch (imaging), or its day-2 transcriptome (LARRY) |
| action | the culture condition: cytokine (G-CSF, M-CSF, ...), or a drug/mutation switch |
| next state | the cell's (or clone's) later appearance or fate: N vs M, division, death |

To learn the action term, the data must show **the same kind of starting cell under different
actions**. LARRY's cytokine experiment does that (sister cells split across 5 conditions).
None of the public time-lapse sets found so far has several cytokine conditions.

## Candidate data (searched 2026-11-04)

### Imaging (time-lapse, fate-labelled)

| dataset | what | access | fit |
|---|---|---|---|
| **Buggenthin 2017 exemplary data** (Helmholtz Nefeli, doi via nmeth.4182) | **Checked 2026-11-04: a demo, not a training set.** 1,536 brightfield PNG frames from 2 positions of experiment 3, backgrounds, and `demoTracks.mat` with **2 tracked cells** (cellNr 82 and 96, 810 time points, marker channel and a 0/1 type label) | open, CC-BY 4.0, 1.1 GB; downloaded and md5-verified to `data/imaging/buggenthin2017/buggenthin2017_try1.zip` | enough to build and test a pipeline (patch extraction along tracks, frozen-DINO features), **not** to measure fate prediction; the full set (thousands of tracked cells) is not public |
| HSPC2 (Zenodo 22163254) | 421 brightfield images of human HSPCs with instance masks | open, 381 MB | segmentation only; no tracks or fates |
| RPE1 FUCCI time-lapse (Zenodo 14774038) | 5,188 full cell-cycle tracks, brightfield + H2B, cell-cycle labels | open | pipeline testbed for a next-state predictor (cell cycle as "fate"); not haematopoiesis |
| Schroeder lab 2025 (Nat Commun s41467-025-57096-y), Moussy 2017, Rieger 2009, Endele 2017 | tracked HSPC / GMP movies, some under G-CSF / M-CSF | not found as public downloads; "on request" = out of scope | would be ideal (cytokine as action), but not available |

### Lineage-traced single-cell transcriptomes (like LARRY)

scLTdb (scltdb.com, NAR 2025) curates 109 datasets, 37 of them haematopoietic, with processed
h5ad/rds files. Those closest to our design:

| dataset | design | "action"? | use |
|---|---|---|---|
| Weinreb 2020 cytokine (GSE140802) | in vitro, sister split over 5 cytokines | **yes** | our current data |
| Weinreb 2020 in vitro time course and in vivo (GSE140802) | same LARRY library, no cytokine split | no | state-fate replication of day-2 priming (B11/B13-style) |
| Rodriguez-Fraticelli 2020 (GSE134242) | LARRY HSCs, transplant, 90 samples | no | HSC-level priming |
| Jindal 2023 CellTag-multi LSK (GSE216521/GSE216606) | in vitro LSK culture, RNA + ATAC lineage | no | chromatin priming of G vs M |
| STRACK, Cell Stem Cell 2025 (GSE266232, GSE282755) | HSC clones in 27-day culture, with/without conditional leukaemia alleles | **yes** (mutation switch) | action = mutation; same clones with and without |
| Bowling 2020 CARLIN (GSE146972) | in vivo, 5-FU vs control | yes (drug), in vivo | perturbation response, coarse |
| Li 2023 DARLIN (GSE222486), Pei 2020 PolyExpress, Wojtowicz 2023, Weng 2024 ReDeeM (human) | in vivo barcoding | no | breadth only |

### Encoders

- DINOv2 (natural images), frozen: the DINO-WM default.
- scDINO (Pfaendler et al. 2023, github.com/JacobHanimann/scDINO): ViT-S/16 trained on
  multi-channel immune-cell images, checkpoints public.
- Cell-DINO (PLoS Comput Biol 2025): DINOv2 on fluorescence microscopy.
- None found trained on brightfield haematopoietic time-lapse. Brightfield is closest to
  natural-image DINOv2, which favours the frozen-DINOv2 baseline.

## Proposed S experiments (each pre-registered in EXPERIMENT_LOG.md before running)

- **S1: frozen-DINO fate probe on Buggenthin.** *Blocked as a test: the public data has only 2
  labelled cells. Can run as a pipeline demo only (no claims).* Use frozen DINOv2 ViT-S/14 features of
  brightfield patches over the generations before marker onset. A linear or small temporal
  model predicts the later lineage. Benchmark: their CNN+RNN AUC by generation. Question: does
  a generic frozen encoder see lineage choice as early as a trained CNN? CPU is enough
  (small patches); Actions if it is slow.
- **S2: transcriptome action predictor on LARRY.** Inputs are a clone's day-2 state (PCA of
  day-2 cells) plus a condition one-hot (the action); the output is the clone's N/M counts
  per condition. Same train/test discipline as the E series (fit set only, held-out shard
  untouched). Question: does a flexible learned predictor beat E38 on the fit-set statistics?
  If it does, E38 is missing structure. If it doesn't, the mechanistic model is as good as
  the data allow.
- **S3 (only if a multi-condition time-lapse set turns up publicly): a full DINO-WM-style
  image world model with cytokine as action.** Parked until then. Keep the gap-map search
  running at retros for such data.

## Constraints

- No GPU on this PC; C: is nearly full. Data goes on D:, under `data/` (gitignored).
- Downloads are authorised. Licences are recorded per dataset. Nothing is redistributed.

## DINO papers: abstract and conclusions read (2026-11-04)

| paper | what makes it work | scale | lesson for us |
|---|---|---|---|
| DINO (Caron 2021, arXiv 2104.14294) | EMA teacher, centring + sharpening against collapse, multi-crop views, no negatives | ImageNet-1k, 2x8 GPUs x 3 days | the recipe is modality-agnostic, but its views (crops) are image-specific; elsewhere the views must be designed, as our sister cells are |
| DINOv2 (Oquab 2023, 2304.07193) | DINO + iBOT patch loss, KoLeo, curated 142M images, distil to small models | 22k A100-h for ViT-g | for small or specialised data, **start from DINOv2 weights**; do not train from scratch |
| DINOv3 (Siméoni 2025, 2508.10104) | 7B ViT, Gram anchoring keeps dense features from degrading, distilled ViT-S/ConvNeXt | 61k H100-h | the small distilled models are the CPU-feasible option for images |
| DINO-WM (Zhou 2024, 2411.04983) | frozen DINOv2 + transition model on patch features, action MLP per token, CEM planning | offline simulated trajectories | needs **known actions and state-action coverage**; this is exactly why S3 needs multi-cytokine time-lapse |
| DINO-world (Baldassarre 2025, 2507.19468) | frozen DINOv2 + cross-attention predictor, timestamps as RoPE, action blocks added later | 66M videos; H100 nodes | irregular time steps suit time-lapse; action conditioning can be fine-tuned on little data **after** large action-free pretraining |
| scDINO (Pfaendler 2023, bioRxiv) | DINO on five-channel immune-cell microscopy | abstract only; full text rate-limited (429) | off-the-shelf ss-ViTs already separate stem-cell states |
| Cell-DINO (PLoS CB 2025) | DINOv2 with channel-aware augmentations (no blur/solarise), init from DINOv2 | 0.2-5M cell images, 32-64 V100s | gains are at large unlabelled scale; batch effects remain |

**Net for the S series.** Every DINO success relies on large unlabelled data or on pretrained
weights, plus views that preserve the signal. S2's tie between DINO-sister and PCA (1,127 clones,
mostly one day-2 cell each) is what these papers predict at our scale. Two practical routes:
(1) images: frozen pretrained DINOv2/v3-small features, no training; (2) transcriptomes:
pretrain on all LARRY cells (about 130k, every time point; unlabelled, so the held-out shard's
clone labels are not used), then fit only the action head on the fit set (the DINO-world
recipe). Route 2 is the candidate S4.
