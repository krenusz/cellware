# Discovery labeller: Raj gastruloid Dropbox (2026-11-12)

Inventory only. Cellware is PAUSED; no data was opened, no statistic was computed, and no hold-out was spent.

## Files

| file | what |
|---|---|
| `LABELLER_BRIEF.md` | the rules and label schema the agents followed (inherited principles from program.md) |
| `raj_crawl.py` | metadata crawl of the share (path, Dropbox file id, size, date); downloads nothing |
| `raj_tree.jsonl` | crawl output: **16,044 files, 262.7 GB**, 405 folders. The 2026-11-07 map had 12,076 files |
| `raj_digest.py`, `raj_digest.tsv`, `digest_area*.tsv` | per-folder digest; split into the three agents' areas |
| `rules/area1_spatial.json` | seqFISH, images, scripts, figures (62 rules) |
| `rules/area2_seq.json` | ATAC / RNA sequencing and processed outputs (106 rules) |
| `rules/area3_tables.json` | raw tables and annotations (66 rules) |
| `rules/zz_review.json` | my review overrides (applied last) |
| `raj_label.py` | merge: applies the rules (broad defaults first, then specific), writes `raj_labels.tsv` |
| `raj_labels.tsv` | **one row per file, keyed by Dropbox id**: kind, seal, local, usefulness, questions, units, basis, note, rule |
| `label_summary.tsv` | counts and GB per label and folder |

## Result

| label | files | GB | meaning |
|---|---|---|---|
| A | 66 | <0.1 | could serve an open question, new |
| B | 509 | 42.3 | supporting / context |
| C | 469 | 4.5 | replication only |
| D | 7,670 | 114.0 | no analytic use (motif dumps, figures, duplicates) |
| X | 7,330 | 101.8 | sealed / locked, reserved |

All 16,044 files are labelled (0 unmatched). Every label's basis is `metadata`: no file was opened.

## The A items (both seqFISH, both unverified)

1. **`extracted_data/SG_kit1/gastruloid1..27(.csv, _spatial.csv)`** (52 files):
   - per-gastruloid seqFISH tables with spatial companions;
   - new since the 2026-11-07 map;
   - could give clone-resolved expression and position in mixtures (instruction vs selection, domain sizing).
   - **Needs a header check** for a per-cell clone identity before it counts.
2. **`raw_data/SG/08182025` and `08212025` h5ad matrices** (14 files):
   - the latest seqFISH runs, with a green-clone channel and a T/Igfbp5 panel;
   - could support a per-cell clone vs non-clone comparison within one chimera.
   - **Unsure:** only 2-6 ROIs per condition.

## Review changes

- **Five raw ratio-series linescans moved from A to B.** They are the per-gastruloid data behind the mean
  curves used for discovery #21 (G1b): the same gastruloids, so not an independent test.
- **Merge order fixed.** One area's `raw_data/*` catch-all had overridden the other areas' specific rules. Broad
  defaults now apply first.
- **Locked folders checked: all X.**
  - morph_08312026, T_Igfbp5_09022026, clone_bulk_chimera_08232026, Line Scans;
  - the revision data CSVs.

## Treated as sealed when unsure (agents' notes)

- Nimbus JSON dates that share gastruloids with sealed series.
- Differentiation Assay replicates.
- ATAC AvBulk / AvP / PvBulk tables.
- Tobias p3 items.
- ATAC samples V37 / V47 / V48 / V61, which are absent from the sample sheet.

## Next (on resume, Bence's call)

1. **Header-only check of SG_kit1:** download 1 table, read the header row only, and see if a clone id column
   exists. This decides whether A stands.
2. If it stands: H19 counts (cells per clone per gastruloid), then seal by gastruloid before any statistic.
