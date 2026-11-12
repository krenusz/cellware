# Discovery labeller: brief for labelling agents (Raj gastruloid Dropbox)

**Goal:** label every file in the Raj lab "Paper" Dropbox (Ayyappan et al. 2025 data, about 12k files) by how
useful it is for **new** discovery. Labels are stored against the Dropbox file id and path, in
`labeller/raj_labels.tsv`.

**Status:** Cellware is PAUSED. Labelling is inventory, not an experiment. It spends no hold-out and runs no
test.

## Inherited principles (hard rules, from program.md H1-H19 and the standing rules)

1. **No statistic on any data.** You may read file metadata (path, size, date), file names, and the **header
   row and row count** of a file that is already on disk and is **not** sealed or locked. Never compute
   means, correlations, group comparisons, or look at outcome columns.
2. **Sealed / locked / spent files:** label from metadata and DATA_MAP_RAJ.md only. Never open them.
   - Check `experiments/sealed.py` (SEALED_FILES, partitions, HELDOUT_READERS) and the "locked" / "sealed"
     rows of DATA_MAP_RAJ.md.
   - When unsure, treat a file as sealed (H10: ask when unsure; here, the default is sealed).
3. **Do not download anything.** Metadata comes from the crawl (`raj_tree.jsonl`). Local copies are under
   `D:\projects\cellware\data\gastruloid_raj\`.
4. **No outreach:** never contact the authors or anyone else, and never draft contact.
5. **Do not read the authors' results** in figures, READMEs of revision analyses, or result CSVs (for example
   `separation_metrics.csv`, `geo_metrics.csv`). Their existence and schema may be noted from DATA_MAP.
6. **Evidence discipline:** every label states its basis (`metadata` or `header`). Do not claim content you did
   not see.
7. **Scope:** usefulness is judged against the open questions below, not against curiosity.

## Open questions (what "useful" means)

| code | question | status in Cellware / literature |
|---|---|---|
| Q1 | instruction vs selection: does a condition change a clone's fate, or select clones already biased? | open in the literature for gastruloids; our LARRY method exists |
| Q2 | clone memory / heritability over divisions and passages | ch5 parked: needs >= 10 clones x >= 3 passages |
| Q3 | early-state readout: does an early molecular state predict a clone's later fate? | ch7-8 confirmed in other systems; gastruloid ATAC/RNA not tested on sealed data |
| Q4 | spatial: how clone states set the axis / domain sizes in mixtures | ch6 closed exploratory; pre-empted mechanism (bioRxiv 2025) |
| Q5 | carrier: is the clone state carried in chromatin, RNA, or both? | ch7 reopen trigger: same-sample RNA + ATAC on clones |

## Labels (one row per file)

| column | values |
|---|---|
| `id` | Dropbox file id (`id:...`) |
| `path` | path in the share |
| `bytes` | size |
| `kind` | raw_seq, alignment, peaks, motif, counts, table, annotation, image, movie, script, figure, object, other |
| `seal` | sealed, locked, spent, open, new (= never touched by Cellware) |
| `local` | yes / no |
| `usefulness` | **A** = could serve an open question with enough units, new; **B** = supporting or context for an A item; **C** = only replicates known results; **D** = no analytic use (figures, motif dumps, duplicates, logs); **X** = sealed or locked, use reserved |
| `questions` | Q1-Q5, comma-separated, or `-` |
| `units` | the analysis unit and an approximate count if visible from names or headers (for example `gastruloid ~20`, `sample 18`) |
| `basis` | metadata / header |
| `note` | one short line: why, and what a test would need |

**Grouping is fine:** label a folder pattern once and apply it to all matching files, unless a file differs.
Output TSV rows must still be one per file.
