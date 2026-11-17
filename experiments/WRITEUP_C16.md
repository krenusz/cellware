# C16 write-up: a hidden inherited state for paraxial vs lateral plate on embryo lineage trees

**Status: closed exploratory, 2026-11-17. No claim.** Train embryos only (E7.5-R1, E8.0-R1, E8.0-R3, E8.5-R2, E8.5-R3, E9.0-R1, E9.5-R1; E7.5 has no P/L leaves, so 6 carry a statistic). Partition A was spent in C14. B stays sealed: it has one usable embryo (E9.0-R3), and the oracle's power gate refuses a look with fewer than 2 units (H14). C stays sealed. All work by Opus 5.5.

## Question
Do lineage relatives share a hidden inherited bias towards paraxial (P) or lateral-plate (L) mesoderm, beyond time, axial identity and position, and is it visible in expression?

## Results (per embryo, leave-one-embryo-out where models are fitted)
| Step | Test | Result | Run |
|---|---|---|---|
| S0 | 4-state model with hidden z vs 2-state CTMC (Felsenstein pruning) | Held-out gain +0.037 per leaf, 6/6; half-life about 12 edges | 44082919451 |
| S1 | Same, with depth-varying rates in both models | Gain +0.015 per leaf, 6/6; half-life 10.3 edges; P/L switching slows about 5x with depth; most of the S0 E8.0 gain was time | 44128234610 |
| S2 | Leave-self-out posterior z vs expression PCs within each label | Mean delta rho +0.097, 5/6; E8.0 about 0 | 44113377868 |
| S3a | S2 with expression-inferred axial identity in the base | +0.104, 5/6; axial identity alone adds about 0 | 44126587693 |
| S3b | z predicts own P/L beyond own and relatives' axial identity | Delta AUC +0.374, 6/6 (E8.0 base reversed; E8.5-E9.5 +0.28 to +0.32) | 44126587693 |
| S3c-i | Beyond position fitted within each embryo | E8.0 position AUC 0.74-0.79, family adds +0.05/+0.10; later +0.27 to +0.35, 6/6 | 44139047485 |
| S3c-ii | Relatives' agreement at matched family size and distance | AUC 0.77-0.78 (E8.0), 0.93-0.95 (E8.5), 0.98-0.99 (E9.0-9.5); not tree size | 44139047485 |
| S4a | Power for a one-look confirmation on B | K1 0.98, K2 0.90 (enter); K3 0.58 (does not) | local arithmetic on committed results |
| S4b | Trace beyond own P/L likeness (a) and orthogonal to the fate axis (d) | (a) +0.030, 5/6; (d) +0.081, 6/6, B power 0.79; trace genes differ from P/L genes (Jaccard 0.04-0.12) | 44158644502 |

## Reading
- Before about E8.0, P/L is open and within-embryo position dominates. Between E8.0 and E8.5, families settle on one fate and switching slows.
- The inherited signal in expression looks like shared history: retinoic-acid response with the Meis/Pbx/Hox axial code on one side, Fgf8 and cholesterol-synthesis genes on the other. This fits the Fgf8 decay clock and the Hox timing of ingression: relatives that left the posterior growth zone together carry similar clock readings. Not a dedicated memory programme. This is an interpretation, not tested.
- Position here is expression-inferred axial identity (classifiers on an E13.5 reference), not measured place. The medial-lateral axis is not measured at all.

## Already published (lit/c16_novelty_2026-11-17.md)
The biology is largely known, much of it on the same data set: heritable lineage-associated expression programmes and joint control of fate by lineage and position (Colgan 2026), asynchronous fate restriction (Liu 2026), timing of streak exit (Kinder 1999). What remains distinct is methodological: a hidden-state CTMC with a time control on whole-embryo trees, and the per-embryo separation of family bias from inferred position.

## Limits
- 6 embryos, 2 per stage at most; stage trends are descriptive per embryo.
- Tree reconstruction error not modelled (Pilarski 2026 caveat).
- Synthetic smoke thresholds were loosened four times before real-data runs (logged amendments).
- Snippet-level exposure to the data-source paper's aggregate results on 2026-11-17 (see the novelty file).

## Files
Scripts c16s0_hidden.py, c16s1_time.py, c16s2_link.py, c16s3_position.py, c16s3c_e8.py, c16s4a_power.py, c16s4b_trace.py; results in results_c16/; log in EXPERIMENT_LOG.md; literature in lit/c16_map_2026-11-17.md and lit/c16_novelty_2026-11-17.md.
