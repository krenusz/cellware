# QS3 novelty check (H25, 2026-11-26)

- Searched: the frozen claim-shaped queries from QS3 stage 2, run by the Haiku lookup lane on OpenAlex (15 per query, relevance) and PubMed E-utilities (up to 15 per query), 2026-11-26, returned unfiltered.
  - For each of the 63 hits I read the first ~400 characters of the stored 700-character opening. Full abstracts were read for 19 hits in two more lookups, each with its DOI checked against the PMID.
  - RF1 and RF2 used no queries: the source settled them at claims level (below).
  - The first run's PubMed DOI field did not match its PMIDs, so PubMed hits are listed by PMID.
  - Hit IDs, unfiltered:
  - Q1 "clone size depends on fate neural progenitor lineage barcode": OpenAlex 15, PubMed 0. doi 10.1016/j.cell.2025.04.026; doi 10.1080/28361512.2025.2599802; doi 10.1073/pnas.2000238117; OpenAlex (no DOI); doi 10.15252/embr.202050932; doi 10.1016/j.ccell.2023.03.009; doi 10.1101/2024.10.28.620670; doi 10.1101/2024.07.24.604840; doi 10.1016/j.devcel.2011.07.019; doi 10.1007/s00018-020-03689-3; doi 10.1242/dev.193193; doi 10.1038/s41467-026-70375-6; doi 10.1002/advs.202518084; doi 10.1038/s41467-018-05740-1; doi 10.1101/2024.11.29.625985.
  - Q2 "fate-coupled clone expansion cortical development": OpenAlex 15, PubMed 15. doi 10.1016/j.neuron.2010.03.019; doi 10.3389/fncel.2013.00141; doi 10.1016/s0896-6273(01)00214-8; doi 10.1523/jneurosci.20-10-03725.2000; doi 10.64898/2026.10.01.753275; doi 10.1016/s0896-6273(03)00365-9; doi 10.3389/fncel.2019.00576; doi 10.1093/cercor/bhm238; doi 10.70675/de756d18z5dacz40fdz81d6zf32c06eefd64; doi 10.1038/mp.2017.236; doi 10.1523/jneurosci.4077-13.2014; doi 10.1101/cshperspect.a022392; doi 10.1093/cercor/13.6.541-i; doi 10.1007/s00018-021-04077-1; doi 10.1523/jneurosci.2319-13.2014; PMID 41193842; PMID 41435817; PMID 12946047; PMID 29114004; PMID 31611775; PMID 22538609; PMID 31239245; PMID 31653848; PMID 10879793; PMID 9862491; PMID 31198941; PMID 38084819; PMID 17156776; PMID 8819334; PMID 31969892.
  - Q3 "clone type size bias progenitor clones": OpenAlex 15, PubMed 3. doi 10.1002/(sici)1096-9861(19971013)387:1<42::aid-cne4>3.0.co;2-n; doi 10.1242/dev.027607; doi 10.1016/j.stem.2018.08.017; doi 10.1182/blood.2020006510; doi 10.1182/blood-2017-07-794362; doi 10.3201/eid2001.130652; doi 10.1038/srep33896; doi 10.7554/elife.41090; doi 10.1182/blood-2017-07-746453; doi 10.1038/nature25168; doi 10.7554/elife.66922; doi 10.1016/j.neuron.2012.06.033; doi 10.1016/j.molcel.2020.04.008; doi 10.1016/j.stem.2018.01.011; doi 10.1038/s41586-022-04786-y; PMID 31610612; PMID 11285251; PMID 9331170.
  - The 2023 thesis without a DOI is OpenAlex W7132977612.
- Source papers:
  - rf2020_clone: doi 10.1038/s41586-020-2503-6. Claims level, read 2026-11-18 (lit/c17_source_claims_2026-11-18.md section 3), plus its abstract (PMID 32669716, read 2026-11-26).
  - cortical_clone: the source in lit/c17_source_claims_2026-11-18.md section 2, claims level. QS3-CO1-M (Bence OK) read the preprint's (PMC12485932) methods subsection headings and one methods sentence; the published version returned HTTP 500. The sentence does not say how clone clusters are assigned.

## Planned findings
- Finding (QS3-RF1): a clone's stem-compartment fraction agrees between the two recipients it was split into, beyond clone size.
  - Verdict: partly published (in kind)
  - Papers: the source states a split into two secondary recipients, low-output clones, and potency as intrinsic and heritable through transplantation. Its abstract states a clone-intrinsic programme behind functional stem cell heterogeneity. Earlier C17-S1 reads: expansion and frequency agreement across split recipients (the source), and repopulation patterns kept on serial transplantation (lit/novelty/C17-S1.md). Not read: how the source defines output, or whether its split comparison measured the stem/progenitor balance.
  - Classic search: not run; the source settles the class.
- Finding (QS3-RF2): within a recipient, the stem fraction is overdispersed beyond binomial sampling given clone size.
  - Verdict: partly published (in kind)
  - Papers: the source claims clone-intrinsic heterogeneity of output, with many low-output clones, and lists an equipotent-clone null model in its design. Not read: whether one was tested against the other, or on what readout.
  - Classic search: not run (as RF1).
- Finding (QS3-CO1): in non-targeting cortical clones, clone size depends on clone type beyond donor batch and stage.
  - Verdict: partly published (in kind, other tissue)
  - Papers:
    - doi 10.1038/nature11045: in normal heart development, one later fate lineage comes from about eight cells per animal with clonal dominance, while the earlier wall forms from several dozen patches of variable size.
    - doi 10.64898/2026.10.01.753275 (preprint, regeneration context): clonal expansion is required for reaching a regenerative state.
    - Neither tests size across fate classes. No neural abstract read states it; nearest are doi 10.1016/j.neuron.2012.06.033, doi 10.1038/s41467-019-12791-5 and doi 10.1111/cpr.13587.
  - Classic search: the three frozen queries above; recall weak (abstracts only, no recall gate).

## What is left that is ours
For all three, a held-out test on a sealed set of a claim reported in kind: held for Bence (RETRO_2026-11-26 decision 1). No step opens on them. CO1 would also need its readout condition settled before any count. QS3-CO1-M left it unresolved; the defining sentences were not reached.
