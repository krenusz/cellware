"""Partitioned hold-out (rules H11-H13, program.md).

Each partitioned data set is split by a salted hash of its biological unit into 5 buckets:
buckets 0-1 = train (40%, the only data analysis code may read), buckets 2, 3, 4 = sealed
partitions A, B, C (20% each). One oracle look spends one partition; the last unspent partition
needs Bence's explicit OK. Partitions are fixed before any statistic and never redrawn.

    from partitions import train_only
    df = train_only("gastruloid_gse291244", df, unit_col="unit")
"""

from __future__ import annotations

import functools
import hashlib

PARTITIONED = {
    # "gastruloid_gse291244" (sealed 2026-11-07 by RT well / 10x lane) is VOID: the paper pools and
    # dissociates gastruloids before RT, so an RT well or lane is not a gastruloid and every gastruloid
    # fell into every partition. No statistic was ever computed on it. Replaced by the entries below.
    "gse291244_founders": {
        "salt": "cellware-gse291244-founders-v2",
        "unit": "founder clone cluster (`well_cluster`, authors' Debris-seq clustering; 28 clusters in "
                "multiome/monoclonal_gas/cell_well_cluster_assignment.rds and multiome/"
                "cell_well_cluster_assignment.rds; gastruloid cells and mESC cells of a cluster go together)",
        "sealed_on": "2026-11-07",
        "prior_exposure": "Regalado/Qiu Cell 2026 Fig. 5 read: published DE (Tet1, Meg3/Rian, Peg10 in "
                          "PSC-biased subclones) and fate-bias categories of the 26 clusters with >= 200 cells",
    },
    "raj_diffassay": {
        "fixed": {"1": "train", "2": "A"},
        "files": ["DiffAssay_Oct212024.csv"],  # 2 replicates only: replicate 1 train, replicate 2 = one sealed partition
        "unit": "replicate (DiffAssay_Oct212024.csv: 9 clones x 4 conditions x 4 times x 2 replicates; "
                "too small for 4 hash partitions)",
        "sealed_on": "2026-11-07",
    },
    "raj_wnt": {
        "salt": "cellware-rajwnt-v1",
        "unit": "Annotation ID (one line scan = one gastruloid), Wnt hi/lo files of Apr/May 2025",
        "files": ["clone1_Wnt_04262025.csv", "clone3_Wnt_04262025.csv"],
        "sealed_on": "2026-11-07",
    },
    "raj_morph_0831": {
        "salt": "cellware-rajmorph0831-v1",
        "buckets": 3,  # mixtures only, thirds A/B/C; the pure plate (bulk_5s2_6a.json) is the train data
        "unit": "Nimbus annotation _id (one polygon = one gastruloid = one well) in "
                "morph_08312026/bulk+5s2_bulk+6a_5s2+6a.json (5s2+6a 27, 5s2+bulk 14, 6a+bulk 29)",
        "files": ["bulk+5s2_bulk+6a_5s2+6a.json"],
        "sealed_on": "2026-11-07",
        "prior_exposure": "tags and counts per tag/well only; pure plate unread until C4-S2a pre-registration",
    },
    "raj_atac": {
        "fixed": {"0": "train", "1": "train", "2": "A", "3": "B"},  # forecasting split by passage; B = last (H13)
        "unit": "passage (atac_peaks/Sorted_V*_rg_nodup_*.narrowPeak; sample sheet atac_meta/align.txt: "
                "V# -> clone, replicate, passage)",
        "files": ["atac_peaks/ passages 2-3 (V33-V64), locked per file in sealed.DATASETS"],
        "sealed_on": "2026-11-07",
        "prior_exposure": "sample sheet, file names and sizes; paper's peak totals (p1 44,540; p3 41,593) and "
                          "passage-1 anterior/posterior shared-peak shares",
    },
    "celltag_lsk": {
        "salt": "cellware-celltag-lsk-v1",
        "unit": "clone (`clone.id` in data/ext_celltag_multi/hsc.rna_atac.r1_2_master_v2.csv, 6,822 clones; "
                "all siblings of a clone, d2.5 RNA/ATAC and d5 r1/r2, go together)",
        "files": ["hsc.rna_atac.r1_2_master_v2.csv"],
        "sealed_on": "2026-11-11",
        "prior_exposure": "column names and structural counts (day, assay, sibling counts) only; fate columns unread; "
                          "Jindal 2024 methods read, results/figures not read",
    },
    "celltag_reprog": {
        "salt": "cellware-celltag-reprog-v1",
        "unit": "clone (`clone_id` in data/ext_celltag_multi/b4.rna_atac.r1_2_master_rev1_annotated.csv, 8,050 clones)",
        "files": ["b4.rna_atac.r1_2_master_rev1_annotated.csv"],
        "sealed_on": "2026-11-11",
        "prior_exposure": "as celltag_lsk",
    },
    "chspctrack": {
        "salt": "cellware-chspctrack-v1",
        "unit": "clone (`clone_id` in data/ext_chspctrack/cells_with_clone_info.h5ad obs, 13,700 clones; "
                "human CD34+ single-cell-derived clones, serial half-well sampling d3-d12, 8 cytokine conditions)",
        "files": ["cells_with_clone_info.h5ad", "metacells.h5ad"],
        "sealed_on": "2026-11-11",
        "prior_exposure": "obs column names, cells per clone per day / well / condition only; cell_type and "
                          "pseudotime / score columns unread; Blood 2026 abstract read (clones diversify "
                          "stochastically, cytokines bias; clonal memory stabilises a few genes per clone)",
    },
    "resistrace": {
        "salt": "cellware-resistrace-v1",
        "unit": "pre-treatment lineage in GSE223003 processed cell_info (data/ext_resistrace/processed): key "
                "'<arm>_<rep>:s<sisters>' for cells with a sister group, else '<arm>_<rep>:c<cell barcode>'; "
                "only cells with a detected label (drugSens not missing). Post-treatment cells inherit the "
                "partition of their pre lineage through prer_r_group (outcome-linked; join only in train or oracle)",
        "files": ["16 *_cell_info.tsv.gz (8 pre, 8 post), sha256 in data/ext_resistrace/SHA256SUMS_processed"],
        "sealed_on": "2026-11-13",
        "prior_exposure": "column names; per-sample cell counts; via a 2026-11-12 column-shift bug, per-sample number "
                          "aggregates: distinct prer_r_group IDs (pre-resistant families; whether singletons were included is unverified), cells carrying prer_r_group (perfect-match pre-resistant cells), family size spread, and post cells carrying prer_r_group (matched post cells); all pooled across partitions (outcome aggregates) "
                          "and two example rows; no per-lineage outcome; Dai 2024 full text read "
                          "(design, label accuracy, subclone A enrichment reported in the paper)",
    },
    "lsc_larry": {
        "salt": "cellware-lsc-larry-v1",
        "unit": "clone: connected component of LARRY barcodes (Custom features) co-called (>= 3 UMI) in >= 2 D20 ex vivo "
                "cells; key = smallest barcode ID (data/ext_lsc_larry/clone_map_d20.tsv). Barcodes never called at D20 "
                "ex vivo are keyed by their own ID. All cells of a clone in every sample (D20/D40, ex vivo/in vivo) go together",
        "files": ["15 Zenodo 18444428 matrices, sha256 in data/ext_lsc_larry/SHA256SUMS", "clone_map_d20.tsv"],
        "sealed_on": "2026-11-14",
        "prior_exposure": "C12-S0: feature/cell structure; per-partition D20 unit counts (>= 3 cells); pooled assigned/ambiguous cell counts per sample; TRAIN-only presence/size of D20 clones in later samples; preprint design text (1,375 clones >= 3 cells "
                          "ex vivo, 160 in vivo, ~13 clones/mouse in chemo); no figures/outcomes read; QS1-L2-P1 (2026-11-24): the source abstract via the preprint-server API, "
                          "claims level (it states that clonal programs are linked with ex vivo self-renewal, in vivo initiation and treatment response); "
                          "full text not reached (HTTP 520); QS1-L2-P1 check 2 retry 3 (2026-11-26, Sonnet lane): full text read "
                          "at claims level (46 headings, 9 figure titles, 7 matched sentences): design split of D20 cells into "
                          "an in vivo arm and a D40 culture arm; a legend lists a clone-size correlation matrix across D20, "
                          "in vivo and D40 (panel not read); D40 state linked to in vivo treatment response; no sentence "
                          "relating D40 persistence to in vivo presence",
    },
    "petracer_embryo": {
        "salt": "cellware-petracer-embryo-v1",
        "unit": "embryo (GEO GSE342607 / Zenodo 19892784 embryo id, e.g. 'E8.5-R2'; 16 embryos E7.5-E10.0). All cells, "
                "trees and characters of an embryo go together. First salt kept (assignment from public names, before any "
                "data read): train E7.5-R1 E8.0-R1 E8.0-R3 E8.5-R2 E8.5-R3 E9.0-R1 E9.5-R1; A E8.0-R2 E8.5-R1 E9.0-R2 "
                "E9.5-R2 E9.5-R3; B E7.5-R2 E7.5-R3 E9.0-R3; C E10.0-R1 (one embryo: weak last reserve)",
        "files": ["4 Zenodo 19892784 files (trees.h5td, cells.csv, characters.csv, spatial_scores.csv), "
                  "sha256 in data/ext_petracer/SHA256SUMS"],
        "sealed_on": "2026-11-15",
        "prior_exposure": "C13-S0: GEO/Zenodo metadata only (sample titles, stages, file names and sizes); abstract",
    },
    "growth12_host": {  # A/B/C VOID 2026-11-17 (amendment 1); train identical to v2
        "salt": "cellware-growth12-host-v1",
        "unit": "host (leading number of the unit ID `<host>_<arm>_T<n>` in Zenodo 5847461, C15 candidate #12; all "
                "growths, cells and trees of a host go together; 32 hosts, 78 units, no host spans arms). Rule fixed "
                "2026-11-17 before any host ID was seen in clear (C14-S5c output was recoded)",
        "files": ["Zenodo 5847461 tarball (1,305 MB, md5 checked vs the record on each runner download)"],
        "sealed_on": "2026-11-17",
        "prior_exposure": "paper abstract and methods (topic level); recoded structure: 78 units, 32 hosts, 3 arms, "
                          "units per host, rows per unit from an allele-like table (C14-S5c)",
    },
    "growth12_host_v2": {
        # C15-S0 amendment 1 (Bence 2026-11-17: "yes rebalance"): train = v1 train; the 15 v1-sealed hosts dealt
        # by leaf count (snake ABCCBA) to 5/5/5. Structure only was known. Use this, not growth12_host, for A/B/C.
        "fixed": {"3430": "A", "3432": "B", "3433": "train", "3434": "B", "3435": "train", "3451": "A", "3454": "train", "3457": "A", "3458": "C", "3460": "train", "3464": "train", "3465": "train", "3466": "train", "3467": "train", "3505": "train", "3507": "A", "3508": "C", "3510": "B", "3511": "C", "3513": "train", "3515": "train", "3519": "train", "3520": "train", "3522": "B", "3703": "train", "3724": "A", "3726": "C", "3730": "train", "3732": "C", "3754": "train", "3765": "train", "3777": "B"},
        "unit": "host number (see growth12_host)",
        "sealed_on": "2026-11-17",
        "prior_exposure": "as growth12_host, plus leaves per host per partition (c15s0_rebal.json)",
    },
    "rf2020_clone": {
        "salt": "cellware-rf2020-clone-v1",
        "unit": "clone: '<sample set>:<LARRY barcode set>' in GSE134242 (all cells of a clone, both recipients and "
                "compartments, go together); strata sample set x shared-across-recipients",
        "fixed_file": "partition_maps/partition_rf2020_clone.tsv",
        "files": ["GSE134242 Sample1-5 barcode_list / samp_id(_flat), sha256 in results_c17/c17s0c_seal.json"],
        "sealed_on": "2026-11-18",
        "prior_exposure": "C17-S0b structure counts (recipients, shared clones per sample set); C17-S0c partition "
                          "counts; claims-level read of the source paper (lit/c17_source_claims_2026-11-18.md); one pooled "
                          "2020 result sentence on across-recipient clone-expansion correlation (lit/novelty/C17-S0.md); "
                          "CAND-F-P1: the 2026 source's Fig. 5f mouse ids and clone counts only (bias columns unread); "
                          "per-recipient clone counts under 4 filters, all partitions pooled (results_cand/candf_p1_struct.json)",
    },
    "cortical_clone": {
        "salt": "cellware-cortical-clone-v1",
        "unit": "clone: obs row of GSE284197 clones.h5ad; strata guide target x donor batch",
        "fixed_file": "partition_maps/partition_cortical_clone.tsv",
        "files": ["GSE284197_clones.h5ad, sha256 in results_c17/c17s0c_seal.json"],
        "sealed_on": "2026-11-18",
        "prior_exposure": "C17-S0b obs structure (columns, levels, guide and donor labels); C17-S0c partition counts; "
                          "claims-level read of the source paper",
    },
    "crest_split_clone": {
        "salt": "cellware-crest-split-clone-v1",
        "unit": "split clone of GSE210139 (C17-S1C amendment 2): '<embryo k>:<informative pattern.v2>' present in the "
                "in vivo arm (pandaE11_rep k) and the organoid arm (OGN_rep k) and in no other sample; strata = embryo",
        "fixed_file": "partition_maps/partition_crest_split_clone.tsv",
        "files": ["GSE210139 single-cell barcode QC tables (rds.gz), sha256 in data/ext_crest/raw/SHA256SUMS"],
        "sealed_on": "2026-11-19",
        "prior_exposure": "C17-S1C schema (column names/types; the cell type column was not tabulated), barcode "
                          "pattern sharing between samples, cells per arm per clone, partition counts; GEO design text "
                          "and the source abstract (claims level; methods paywalled)",
    },
    "dtt_founder": {
        "salt": "cellware-dtt-founder-v1",
        "unit": "founder clade of the DNA Typewriter mouse dated tree (C17-S1PL): maximal clade with <= 20,000 leaves, "
                "'F<node index in preorder parse of merged_minB2h_lineage_constrained.nwk>'; strata = first-cleavage half",
        "fixed_file": "partition_maps/partition_dtt_founder.tsv",
        "files": ["shendurelab/dtt-mouse v1.1.0 tree + cell metadata, sha256 pinned in c17s1pl_tree.py"],
        "sealed_on": "2026-11-20",
        "prior_exposure": "C17-S1PL schema (column names/types; only cell_id examples), tree topology, branch lengths, "
                          "leaf-ID join, pair and founder counts (gate 1 and 2); source abstract (claims level)",
    },
    "germline_fish": {
        # H24 map from candi_seal.py, run 45222067549: eligible fish (>= 6 sample indices) 8 train, 4/4/4 sealed
        "fixed": {"1": "train", "2": "train", "4": "A", "5": "train", "8": "train", "10": "A", "12": "train", "13": "A", "14": "B", "15": "C", "16": "train", "17": "C", "18": "train", "22": "A", "26": "B", "35": "B", "37": "B", "38": "A", "40": "train", "42": "train", "44": "C", "45": "train", "50": "C", "51": "B"},
        "unit": "numbered fish in Gagnon-lab/weber-shrader at commit 99b327c (CAND-I). The seal holds for that fish "
                "number in every pipeline variant, table, matrix and summary. Three control fish are outside the map "
                "and never read. Heatmap images and README figures stay unopened",
        "files": ["source tables at the pinned commit, downloaded at run time, never copied into this repo"],
        "sealed_on": "2026-11-23",
        "prior_exposure": "source text at claims and methods level, including its stated median drift estimate; column "
                          "names and sample counts (CAND-I-P1); repository tree with file names and sizes, including "
                          "per-fish matrix sizes; sample indices per partition",
    },
    "tls_tracer": {
        # H24 map frozen from c17p10_s0.py run 45405214801 (drawn from cell counts before any tree code; C17-S1R-P10)
        "fixed": {"TLS1": "C", "TLS2": "B", "pool:Bar1": "A", "pool:Bar10": "C", "pool:Bar11": "C", "pool:Bar12": "A",
                  "pool:Bar13": "C", "pool:Bar14": "train", "pool:Bar15": "train", "pool:Bar16": "train",
                  "pool:Bar18": "train", "pool:Bar19": "B", "pool:Bar2": "train", "pool:Bar20": "B", "pool:Bar21": "B",
                  "pool:Bar22": "train", "pool:Bar23": "C", "pool:Bar24": "A", "pool:Bar3": "train", "pool:Bar4": "B",
                  "pool:Bar5": "A", "pool:Bar6": "train", "pool:Bar7": "train", "pool:Bar8": "A", "pool:Bar9": "train"},
        "unit": "one trunk-like structure in GSE220949: the single-structure samples TLS1 (GSM7744222) and TLS2 "
                "(GSM7744223), or one MULTI-seq call Bar1-Bar24 of the 24-structure pool (GSM7744224 alleles joined to "
                "GSM7744226 calls; Bar17 has no call). Cells without a call have no unit and are dropped before "
                "partition_of. Explant pools, the ESC time course and all scRNA archives are outside the map",
        "files": ["GEO GSE220949 allele tables and MULTI-seq calls, downloaded at run time, never copied into this repo"],
        "sealed_on": "2026-11-24",
        "prior_exposure": "GEO series and sample metadata; column headers; cell-id columns and a whole-table tally of "
                          "r1-r3 allele tokens of the pool (no fate label); explant-pool-1 call counts; per-unit cell "
                          "counts; train-only tree pair counts, token tallies and recurrence (C17-S1R-P10, runs "
                          "45405214801 void and 45406339352); abstracts of the source (doi 10.1016/j.devcel.2024.03.024) and two "
                          "reanalyses (Carta, doi 10.1038/s41592-025-02903-z; TROUPE, preprint 10.64898/2025.12.09.693214), "
                          "claims level (CAND-Y-P1)",
    },
    "mouse_gast_tree": {
        "salt": "cellware-mouse-gast-tree-v1",
        "unit": "track tree of the S-BIAD1682 manual tracks (DS-EMB-S): connected component of the spot graph, "
                "'M<movie>:<earliest spot ID>', Movies 1-4; Movie5 repeats Movie3's tracking and is outside the map "
                "(never read); strata = movie, size = eligible depth >= 1 sister pairs",
        "fixed_file": "partition_maps/partition_mouse_gast_tree.tsv",
        "files": ["BioStudies S-BIAD1682 MaMuT tracking XML, sha256 in data/ext_mouse_gast_tracks/raw/SHA256SUMS"],
        "sealed_on": "2026-11-25",
        "prior_exposure": "BioStudies study fields (organism, imaging method, annotation type and method, the "
                          "experimental-variable and image-analysis text, genotype strings in the file list); XML "
                          "feature declarations; spot, edge, division, tree and eligible-pair counts per movie; "
                          "presence counts of the intensity attributes (no values); study title (DS-EMB sweep); "
                          "source abstract and full-text claims (section headings, sentences on sisters, generations "
                          "and fate restriction) of doi 10.1038/s44318-025-00441-0 (DS-EMB-N)",
    },
}
BUCKETS = 5
TRAIN = (0, 1)
SEALED = {"A": 2, "B": 3, "C": 4}


@functools.lru_cache(maxsize=None)
def bucket(dataset, unit):
    salt = PARTITIONED[dataset]["salt"]
    return int(hashlib.sha256(f"{salt}:{unit}".encode()).hexdigest(), 16) % PARTITIONED[dataset].get("buckets", BUCKETS)


@functools.lru_cache(maxsize=None)
def _fixed_from_file(dataset):
    """H24 balanced map stored as a committed TSV (unit<TAB>partition), for maps too large to inline."""
    import os  # noqa: PLC0415

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), PARTITIONED[dataset]["fixed_file"])
    with open(path, encoding="utf-8") as fh:
        return dict(line.rstrip("\n").split("\t") for line in fh if line.strip() and not line.startswith("#"))


def partition_of(dataset, unit):
    fixed = PARTITIONED[dataset].get("fixed")
    if fixed is None and "fixed_file" in PARTITIONED[dataset]:
        fixed = _fixed_from_file(dataset)
    if fixed is not None:
        return fixed[str(unit)]
    b = bucket(dataset, unit)
    if PARTITIONED[dataset].get("buckets") == 3:
        return "ABC"[b]
    if b in TRAIN:
        return "train"
    return next(k for k, v in SEALED.items() if v == b)


def train_only(dataset, df, unit_col):
    """Rows of df whose unit is in the train partition. The only allowed way to read a
    partitioned data set outside test_oracle."""
    keep = df[unit_col].map(lambda u: partition_of(dataset, u) == "train")
    return df[keep]


def sealed_rows(dataset, df, unit_col, part):
    """Rows of one sealed partition. Call ONLY from an oracle test function (check_sealed enforces)."""
    assert part in SEALED, part
    keep = df[unit_col].map(lambda u: partition_of(dataset, u) == part)
    return df[keep]




BALANCED_ORDER = ("train", "train", "A", "B", "C", "C", "B", "A", "train", "train")  # H24: 40/20/20/20, snake


def balanced_split(sizes, salt, strata=None):
    """H24: balanced sealing. sizes = {unit: structural size}; strata = {unit: label} (optional).
    Within each stratum, units sorted by size (desc; ties by sha256(salt:unit)) are dealt in BALANCED_ORDER,
    continuing the position across strata so totals stay balanced. Returns {unit: partition} to commit as a
    fixed map. Structure only (unit counts, cell/leaf counts, design labels); never outcomes."""
    strata = strata or {u: "" for u in sizes}
    tie = lambda u: hashlib.sha256(f"{salt}:{u}".encode()).hexdigest()  # noqa: E731
    out, i = {}, 0
    for s in sorted(set(strata.values()), key=str):
        for u in sorted((u for u in sizes if strata[u] == s), key=lambda u: (-sizes[u], tie(u))):
            out[u] = BALANCED_ORDER[i % len(BALANCED_ORDER)]
            i += 1
    return out
