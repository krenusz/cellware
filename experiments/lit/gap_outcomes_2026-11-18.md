# Why the C13 gaps led to no discovery (2026-11-18)

Bence: "do a full map, novelty cvhek and tell me why the gaps recognised in this [C13-S1 Lineage x Space Map, 2026-11-15] lead to no fcking discovery". Sources: the map artifact (four open claims), chapters 14-16, RETRO_2026-11-18_FULL, novelty_audit_2026-11-18, and 4 web searches today. All by Opus 5.5.

**Exposure:** today's searches again returned snippets of the embryo source paper (Colgan 2026) on axial progenitors: they stay polyclonal during elongation; bipotent clades give neural and mesodermal descendants on distinct paths; not every axial progenitor is bipotent; an A-P bias appears early. Aggregate claims, no numbers for B or C. Added to the petracer_embryo exposure record.

## The four gaps, one by one
| gap (C13-S1 map) | map said | what happened | status today | why no discovery |
|---|---|---|---|---|
| G1 clone history predicts where cells end up, beyond type (the morphogenesis question) | C13 cannot test: positions inferred | C13 closed on that fact | **still open in whole embryos**; spatial lineage with measured position exists only in colonies and tissue (baseMEMOIR mESC colonies via LineageMap 2026; Spatio-DARLIN 2026 in mouse tissue) | the only real knowledge gap had no public data. We stopped at "no embryo data" and never looked for the question in systems that do have measured position |
| G2 state beyond type predicts clade fate in held-out embryos | testable; source overlap "low" | C14 CONFIRMED on A | **partly published**: the source reports heritable fate programs and origin traces; lineage inpainting 2026 reports late fate separation | the gap was a gap in *testing standard*, not in knowledge. The "low" overlap was a guess with the source unread. A held-out confirmation of what the authors describe is validation, not discovery |
| G3 embryo fate rules replicate out of sample | maybe, power unknown | never run | open as replication | a replication is not a discovery by construction |
| G4 axial progenitors mature towards neural fate in vivo | maybe; source overlap "medium" | never run; C16 took the adjacent P/L question instead | **published** by the source (polyclonal axial progenitors, bipotent clades with distinct paths) | the medium-overlap warning was ignored and the neighbour question was published too |

## The root causes
1. **The gaps were derived from one data set's affordances.** The map asked "what can this data set test?", so every gap was bounded by what the authors had collected. The authors had already mined it. A gap list built that way finds *missing tests*, not *missing knowledge*.
2. **The overlap column was filled while the source was unread.** "Low" and "unknown" were guesses, and both were optimistic. H25 claims-level reading now fixes this.
3. **The one real knowledge gap (G1) needs data that does not exist publicly.** With no wet lab and no outreach, our discoveries can only come from data the authors could not combine: across data sets, across systems, or through a model. No chapter was chosen on that basis.
4. **The gates select for answerable questions.** H19 rewards powered tests on data in hand, and those are the questions the data's owners asked first. Passing the gates was reported as "CONFIRMED" and read as success, so the selection pressure never pointed at novelty until H25.
5. **The gap map ranks papers, not questions.** lit_gap.py scores papers by topic keywords; it surfaces neighbours, not unanswered questions. 26 of 5,747 relevant works are marked read.

## Where a discovery is still possible (each needs its own H25 file before work)
| candidate | why it could be ours | data | first check |
|---|---|---|---|
| **A. Cross-recorder transfer of C14:** the model frozen on PEtracer train embryos predicts relatives' fate in the independent DNA Typewriter mouse embryo (zygote to E13.5, 1.28M cells, different recorder and lab) | neither source paper can do it; no cross-data-set test found | tree public via the NextCell browser; raw accession not yet found (Science 2026, doi 10.1126/science.ael0508) | data availability and format; one embryo, so the unit is clades (H22/H24 check) |
| **B. G1 outside embryos:** does lineage history predict measured position beyond cell type in mESC colonies (baseMEMOIR) or mouse tissue (Spatio-DARLIN)? | the morphogenesis question on data with measured position | baseMEMOIR (public with LineageMap); Spatio-DARLIN (Nat Methods 2026) | claims-level read of both; LineageMap already infers ancestral positions, so overlap may be high |
| **C. Write up the four corrections** (chapters 1-2: instruction size, private small clones, the KO fate effect not identifiable, the control wells not null) | already ours and held on sealed or recovery checks | done | a reanalysis note per data set |

## Novelty check of the candidates (snippet level)
- A: no paper found that transfers a fate model between independent embryo lineage recorders. Not found.
- B: LineageMap 2026 infers spatially resolved trees and ancestral locations (method); the biological claim "history beyond type predicts position" was not found. Partly open.
- C: unchallenged since the 2026-11-12 full-text citer check.
