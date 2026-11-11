# Extended retro: the whole project (2026-11-11, at pause)

Bence: "do a full project recap and extended retro, literature rereview". Scope: chapters 1-9, process,
rules, and how the collaboration went. Companions: PROJECT_RECAP_2026-11-11.md, lit/review_2026-11-11.md.

## 1. Scorecard

| chapter | chosen because | ended | primary test confirmed on sealed data? |
|---|---|---|---|
| 1 LARRY cytokines | Bence's question | answered | yes (several looks) |
| 2 SIS-seq | generalisation | answered, mixed | partly (H yes; RNA trace weak; Bcor not identifiable) |
| 3 gastruloid clones | morphogenesis goal | answered | yes (G1, G6) |
| 4 mixing rescue | interest | data-limited | no (not replicated) |
| 5 clone memory | interest | data-limited | no look (unpowered) |
| 6 cells into space | morphogenesis goal | exploratory | no possible unseen test |
| 7 fate carrier | interest; rescued by reframing | confirmed (reframed) | yes, 5/5 on A |
| 8 fate readout | **H19** | confirmed | yes, 2/2 run (1 refused by gate) |
| 9 | **H19** | no test, paused | n/a |

- **Pattern:** chapters 1-3 used rich, purpose-fit data and answered their questions. Chapters 4-6 chased
  the long-term morphogenesis goal on data too small for it. All three ended without a sealed result. Once
  H19 forced "what can the data power", chapters 7-8 confirmed. But the questions got narrower and less novel
  (lit review: consistent with LineageFlow, FateLimit and the human clonal-fate papers).

## 2. What worked (keep)

1. **Sealed partitions + oracle (H1-H14).**
   - 20 looks, each pre-registered.
   - No look was ever re-run or tuned after the fact.
   - The power gate refused two underpowered looks (T2, and earlier G-series) **without spending them**.
     That is the most valuable piece of infrastructure.
2. **Pre-registration in EXPERIMENT_LOG with Serves / Pre-flight**, plus a scope-guard hook.
   - Drift stopped after H16.
   - Every claim traces to a commit.
3. **Stopping rule for model iterations** (ch1). It ended endless model tweaking.
4. **Fixed-sequence testing with a safe lead test** (ch7-8). It guaranteed a result without spending extra
   error budget.
5. **Honest withdrawal.** C9-S1 was withdrawn on a readout check before anything sealed was read. The
   "not replicated" findings (#16, ch4) were published as such.
6. **Count functions that read no fate column.** Refusals cost zero exposure.

## 3. What went wrong (root causes, not symptoms)

| # | failure | where | root cause | rule now |
|---|---|---|---|---|
| 1 | Hold-out slip (Bence furious) | ch3 (5 unsealed tests) | sealing came after first statistics | H1, H11, M13 |
| 2 | Wrong unit / mislabelled file / unpowered pre-reg | ch4-5 | pre-registered before checking structure | H15 pre-flight |
| 3 | Data-driven side quests | ch3-6 | no chapter question to serve | H16 scope guard |
| 4 | Power checked after building | ch4, 5, 7 | chosen for interest | **H19** |
| 5 | 13 model versions in one step | ch6 | no amendment ceiling | H18 retro after 4 amendments |
| 6 | Plan power used the dataset-wide ratio for a subset | ch8 T2 | count shortcut | H19 amendment (subset structural count) |
| 7 | Degenerate outcome passed the scan | ch9 H3 | power computed without looking at the outcome distribution | readout gate (>= 30 per side, statistic must vary with the model) |
| 8 | Heredoc + `</dev/null` lost 5 log entries | ch7 | shell pitfall | windows-pitfalls memory item 6 |
| 9 | Ambiguous "2" ran a step | ch7 | bare digit with list + buttons | digit-ambiguity rule |
| 10 | K vs T labels confused Bence | ch8 | two numbering schemes | use question names in user-facing text |
| 11 | Novelty never checked at chapter choice | ch7-9 | H19 checks feasibility, not value | **proposed H20** below |

- **Main lesson:** the project's failures moved from *integrity* (1-3), to *feasibility* (4-7), to
  *value* (11).
  - Integrity is now solved by machinery.
  - Feasibility is mostly solved by H19 + the gates.
  - Value is not addressed. H19 selects for tests that confirm what is already known (ch8 overlaps the source
    paper).

## 4. Rule audit (19 rules)

- **Load-bearing:**
  - H1-H4 (sealing);
  - H6-H7 (one look, spent is spent);
  - H11-H14 (partitions, selection, reserve, power);
  - H15-H16 (pre-flight, scope);
  - H19 (feasibility).
- **Rarely triggered:** H8 labels, H10 ask-when-unsure. Keep both; they are cheap.
- **Overlap:** H14 (power before a look), H17 (power from the model's own prediction) and H19.2 (plan power)
  say similar things at three stages. Merge them into one "power at three gates" rule on resume.
- **Missing:**
  - a value gate (H20 below);
  - the readout gate. It is written in C9-S2 but not yet in program.md.

## 5. Proposed changes for resume (need Bence's OK; not adopted)

- **H20 value gate:** a chapter proposal also states, from abstracts:
  - (a) whether the source paper or a 2024+ paper already reports the answer;
  - (b) what the result would add.
  - If it adds only "disciplined replication", say so up front, and Bence decides.
- **Readout gate into program.md:**
  - >= 30 units on each side for binary splits;
  - the statistic must vary with the model;
  - outcome distribution shown at the scan.
- **Merge H14/H17/H19.2** into one power rule with three gates: scan, selection, oracle.
- **Restart only on new data:** DeepTrack, hESC endoderm, or the human haematopoietic organoid ("hidden
  predictors"), if public. Each could serve the open carrier / instruction questions with real novelty.

## 6. Collaboration

- **Bence's interventions came at three recurring points:**
  - hold-out care (ch3);
  - drift (ch3-6);
  - chapters ending data-limited (ch7).
  - Each became a hard rule, and the same failure did not recur.
- **Friction now comes from reporting:** labels, ambiguous digits, long tables. Shorter replies with
  question names work better on Remote Control.
- **Effort and usage:**
  - 8 of 9 chapters ran within about 10 days.
  - The weekly plan budget is at 78% at pause.
  - The project can resume cheaply, because everything sits in FOCUS.json, the log and the oracle.

## 7. Literature gap (lit_gap.py + Europe PMC re-review)

See lit/review_2026-11-11.md. In short:
- The confirmed results agree with 2025-2026 work and are not new.
- The open questions are exactly where 2026 multiome lineage-tracing preprints are heading. Their data is
  the restart trigger.
