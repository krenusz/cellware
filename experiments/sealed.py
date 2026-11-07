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
    "gastruloid_gse291244": "PARTITIONED 2026-11-07 (partitions.py): train 40% + sealed A, B, C; "
                            "unit = plate + RT well / 10x sample; one look per partition (H11-H13)",
    "camellia": "exploratory only (no confirmation claims without a new sealed shard)",
    "darlin": "exploratory only (no confirmation claims without a new sealed shard)",
    "epiclone": "exploratory only (no confirmation claims without a new sealed shard)",
    "imaging": "exploratory only (no confirmation claims without a new sealed shard)",
    "fm_weights": "model weights, not data",
    "orthologs": "reference table, not data",
    # G9 mechanism set (sealed 2026-11-07 before any statistic; structure check only). Drug series
    # partitioned by experiment series (dates from scan IDs): train = Dec 2024 + Feb/Mar 2025 files;
    # A = May 3 2025, B = July 28 2025, C = May 24 2025 (iou metrics only). Locked = not analysed yet.
    "data/gastruloid_raj/raw_ls2/1+3_ctrl_ra_sb_05032025.csv":
        ("470b61595fbee65c56241fdb511a2a5cf508d2fba6075cc8c06cd9ecbeddfb2b", ["g9_confirm.py"], "sealed drug partition A"),
    "data/gastruloid_raj/linescans/Ctrl_clone1_clone3_050325.csv":
        ("a656940ec9e01923d355cbb5589da3826b7b59e4619ff7fde8abd003a536d2b8", ["g9_confirm.py"], "sealed drug partition A"),
    "data/gastruloid_raj/linescans/RA_clone1_clone3_050325.csv":
        ("63c232806320e90982446459087172aef2d6abe2f6eff8d97fd565c0adcabf68", ["g9_confirm.py"], "sealed drug partition A"),
    "data/gastruloid_raj/linescans/SB_clone1_clone3_050325.csv":
        ("0db3dd9caccda6b2087cf0cf5b4a406b360126f97f432f0497c72de841d56f74", ["g9_confirm.py"], "sealed drug partition A"),
    "data/gastruloid_raj/iou/gastruloid_analysis_results_1+3_05032025.csv":
        ("ea05a57d37bc8f5e07078262c19c11bae6e1e16bdab677ee3f9aaf3ac6f75d20", ["g9_confirm.py"], "sealed drug partition A"),
    "data/gastruloid_raj/raw_linescans/07282025/linescan_drug_07282025.csv":
        ("9bcb7b94c8a51c97212114bb45a3d806391a194b28819f62b8874ef8448ba305", ["g9_confirm.py"], "sealed drug partition B"),
    "data/gastruloid_raj/iou/gastruloid_analysis_results_1+3_07282025.csv":
        ("df090b9fb63ab4ef9670440fe4da99e784e66fe7aa6ca3e988b6a809626a3485", ["g9_confirm.py"], "sealed drug partition B"),
    "data/gastruloid_raj/iou/gastruloid_analysis_results_1+3_05242025.csv":
        ("33021182b4e2f1f3e0ec44572b838573c84263f8cd04a37c8665bc7662fb2c4d", ["g9_confirm.py"], "sealed drug partition C"),
    "data/gastruloid_raj/iou/gastruloid_analysis_results_1+6_05242025.csv":
        ("e155320d42fdd32a2e2d0aa23220da8f3d409559e546a4a996b65861c0c13dde", ["g9_confirm.py"], "sealed drug partition C"),
    "data/gastruloid_raj/iou/AP_CloneFraction_Drug - Sheet1.csv":
        ("ff0da43e8815901506a9dcc6f673b45471dec97d07182a229afc36fc4501517e", [], "locked (series unknown; could overlap a sealed series)"),
    "data/gastruloid_raj/diff_assay/DiffAssay_Oct212024.csv":
        ("cde7a581ca9431de95cec9dcbb47c724495c897a5a325d48808149b3ca1390c4", [], "locked: partition before any statistic (option 3, #21)"),
    "data/gastruloid_raj/raw_linescans/07162025/line_scan_GFPspike.csv":
        ("22e11f6a1519c290659ba854453d81ede6d5a19c92202f13f99b0a8b7ae0126e", [], "locked (design unknown)"),
    "data/gastruloid_raj/raw_linescans/07162025/line_scan_GFP%spike_1.csv":
        ("6f09bfd7c68c30d70644c1e2834c6ba52a6368e038ab9b7ca5d7f755a172f8ba", [], "locked (design unknown)"),
    "data/gastruloid_raj/raw_linescans/08052025/line_scan_sgbc2_50%.csv":
        ("696361308f340a2e7558a77d1ea43addfb207cff31344a8220b64fbdb519d3ea", [], "locked (design unknown)"),
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
    # G7 (sealed 2026-11-06 before any statistic; never read before: names listed only; structure check H9)
    "data/gastruloid_raj/raw_ls2/3+7_6+7_linescan_5032025.csv":
        ("c8d6b96bb0fa1379ff149ec40ba6f3fb3e97e90750658ab6e747a5d153d48ab2", ["g7_confirm.py"], "sealed G7a"),
    "data/gastruloid_raj/raw_ls2/clone1%_pass2_07092024.xlsx":
        ("a4e5ef2cda603f374808b3c8c0a6b4d2395a61cda07cb1c8c29248c603055758", ["g7_confirm.py"], "sealed G7b"),
    "data/gastruloid_raj/raw_ls2/clone1%_pass3_07112024.xlsx":
        ("9348f950373e5b971a15d9a5510dc58cf9973e8fdd89f4df15ee3f6d110c6f04", ["g7_confirm.py"], "sealed G7b"),
    "data/gastruloid_raj/raw_ls2/clone3%_pass2_07092024.xlsx":
        ("85c078982b3891b08bc739f1b7de10e3007dc5a766a5dd721b7c5d977a14054d", ["g7_confirm.py"], "sealed G7b"),
    "data/gastruloid_raj/raw_ls2/clone3%_pass3_07112024.xlsx":
        ("a9f6a9f5b0a6d8b4f255aa3ede837b51e0d7b61b37c7860e3a7319040e3fd979", ["g7_confirm.py"], "sealed G7b"),
    "data/gastruloid_raj/raw_ls2/clone3%_pass4_7132024.xlsx":
        ("58b439d2d61b8561b6710dbd57c38e6cdf1cf6a4f1f94a026d2966f41db14baa", ["g7_confirm.py"], "sealed G7b"),
    "data/gastruloid_raj/raw_ls2/clone3_pass3_07112024.xlsx":
        ("011939d615af3d5ddab28427655b2e455847946f387489822cae38573d1865b2", [], "sealed reserve (untested); LAST Raj sealed data: Bence OK needed (H13)"),
}

# modules allowed to contain held-out access patterns (each reviewed; reason stated)
HELDOUT_READERS = {
    "test_oracle.py": "the oracle",
    "core_sisseq.py": "scores('heldout') only inside heldout_stat (E54)",
    "sis_data.py": "_keep(part='heldout') loader, called only by oracle tests",
    "e55_fm.py": "split() marks held-out clones; heldout_stat (E55)",
    "g1_gastruloid.py": "polarity(include_heldout=True) only in fit (counts) and heldout_stat (G1a)",
    "g6_confirm.py": "G6 sealed-file readers",
    "g7_confirm.py": "G7 sealed-file readers (oracle only)",
    "g9_confirm.py": "G9 sealed drug partitions A/B/C (oracle only)",
    "clone_state_model.py": "split() excludes shard 1 for fitting",
    "b32_small_clones.py": "excludes held-out shard",
    "b36_celltypes.py": "excludes held-out shard",
    "s2_action_predictor.py": "excludes held-out shard",
    "r2_skew_sbi.py": "heldout_index (statistic indices only, no data)",
}
