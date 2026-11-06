"""Registry of held-out (sealed) data. Hold-out rules H1-H8 are in program.md.

Every data set under data/ must have an entry in DATASETS before its first statistic, saying how
it is sealed (or that it is exploratory only and may support no confirmed claim). Sealed files are
listed in SEALED_FILES with their sha256 and the only modules allowed to name them.
`check_sealed.py` enforces this; test_oracle runs it before every look.
"""

# data dir -> how its hold-out is defined
DATASETS = {
    "larry_cytokine": "hash shards via test_oracle.shard_of (epo_scf_pooled; B37 shard 1 of 2) - spent",
    "larry": "hash shard e55:<clone> 0 of 3 (larry_fm_heldout) - spent",
    "larry_invivo": "exploratory only (no confirmation claims without a new sealed shard)",
    "sis_seq": "hash shard H2 0 of 3 (skew_heldout, sisseq_rna_heldout) - spent",
    "gastruloid_raj": "G1a hash shard g1:<gastruloid> 0 of 3 (spent); G6 sealed files (spent); "
                      "all other files: OPEN data, results 'unconfirmed' (M13)",
    "gastruloid_gse291244": "NOT SEALED YET - seal before any statistic (rule H1)",
    "camellia": "exploratory only (no confirmation claims without a new sealed shard)",
    "darlin": "exploratory only (no confirmation claims without a new sealed shard)",
    "epiclone": "exploratory only (no confirmation claims without a new sealed shard)",
    "imaging": "exploratory only (no confirmation claims without a new sealed shard)",
    "fm_weights": "model weights, not data",
    "orthologs": "reference table, not data",
}

# sealed files: path relative to repo root -> (sha256 or None, allowed reader modules, status)
SEALED_FILES = {
    "data/gastruloid_raj/raw_ls2/clone1+bulk_3+bulk_5+bulk_pass4_12042024.xlsx":
        (None, ["g6_confirm.py"], "spent G6a 2026-11-06"),
    "data/gastruloid_raj/raw_ls2/clone1+bulk_3+bulk_5+bulk_6+bulk_pass4_01272025.xlsx":
        (None, ["g6_confirm.py"], "spent G6a 2026-11-06"),
    "data/gastruloid_raj/raw_ls2/clone1+3_pass1_01172024.xlsx":
        (None, ["g6_confirm.py"], "spent G6b 2026-11-06"),
    "data/gastruloid_raj/raw_ls2/clone1+3_clone1+6_pass4_01272024.xlsx":
        (None, ["g6_confirm.py"], "spent G6b 2026-11-06 (tag 1+3); tag 1+6 unread"),
}

# modules allowed to contain held-out access patterns (each reviewed; reason stated)
HELDOUT_READERS = {
    "test_oracle.py": "the oracle",
    "core_sisseq.py": "scores('heldout') only inside heldout_stat (E54)",
    "sis_data.py": "_keep(part='heldout') loader, called only by oracle tests",
    "e55_fm.py": "split() marks held-out clones; heldout_stat (E55)",
    "g1_gastruloid.py": "polarity(include_heldout=True) only in fit (counts) and heldout_stat (G1a)",
    "g6_confirm.py": "G6 sealed-file readers",
    "clone_state_model.py": "split() excludes shard 1 for fitting",
    "b32_small_clones.py": "excludes held-out shard",
    "b36_celltypes.py": "excludes held-out shard",
    "s2_action_predictor.py": "excludes held-out shard",
    "r2_skew_sbi.py": "heldout_index (statistic indices only, no data)",
}
