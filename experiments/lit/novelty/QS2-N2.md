# QS2-N2 novelty check (H25, 2026-11-26)

- Searched: OpenAlex classic search (novelty_search.py functions, run by the Haiku lookup lane from a scratchpad driver), 2026-11-26, three rounds.
  - Round 1 queries: "division orientation inheritance", "spindle orientation daughter cells", "oriented cell division gastrulation", "cell division axis epiblast". Top 25 titles screened per query. All 10 most related abstracts read (7 via OpenAlex, 3 via PubMed).
  - Round 2 (advisor: round 1 mostly probed tissue-level orientation, not the claim) queries: "successive division orientation", "division axis memory", "mother daughter spindle orientation", "consecutive cell divisions orientation correlation", "division plane previous division". Top 25 titles screened per query. 5 more abstracts were checked: 4 were read and 1 had no abstract (doi 10.1007/bf00284884).
  - Round 2's title screen skipped the centrosome, neuroblast and plant hits as off-topic. That was the lookup lane's call; under the any-system rule they are on-topic. The rest stay unread.
  - Round 3 (advisor): 2 abstracts, both read (doi 10.1083/jcb.200905024, doi 10.1083/jcb.105.5.2123).
  - Classing rule fixed before round 2 (EXPERIMENT_LOG, QS2 gate 4 second round): mother-to-daughter division-axis persistence reported in any system makes this "partly published".
- Source paper: doi 10.1038/s44318-025-00441-0, claims level only (as in QS2-N1).
  - Section headings were re-read.
  - A keyword scan of the full text kept 0 sentences where an orientation term (orient, axis, spindle, plane, angle) co-occurs with a division term (sister, mitosis, mitotic, division, daughter). The terms do occur on their own.
  - **Unread:** the sentences holding spindle (3), orient (1) and direction (4). They are to be read at claims level, with the exposure logged, if N2 is ever taken forward.
  - The 20 most-cited references got a title screen only (mostly software and cardiac lineage papers). This is not a full reference pass.

## Planned findings
- Finding: in tracked gastrulating mouse embryos, a daughter's division axis follows its mother's division axis beyond the local tissue orientation (division-orientation memory across generations).
  - Verdict: partly published
  - Papers:
    - **Basis:** fly larval neural stem cells keep the orientation of polarity and division unchanged across consecutive mitoses. The abstract calls this an "orientation memory" that spans the last mitosis and is cell autonomous with respect to the surrounding tissue (doi 10.1083/jcb.200905024).
    - **Basis:** in the early worm embryo, one set of cells divides successively on the same axis and another in an orthogonal pattern, both set by centrosome movements (doi 10.1083/jcb.105.5.2123).
      - The orthogonal pattern is a mother-to-daughter relation that N2's one-sided statistic (|cos| above neighbours) would miss.
    - Mechanism, inferred and not stated as axis persistence: fly male germline stem cells (doi 10.1126/science.1134910, orientation tied to the niche) and fly larval neural stem cells (doi 10.1016/j.devcel.2007.01.021, a retained apical centrosome).
    - Zebrafish neural progenitors in gastrulation: divisions within a clone become progressively oriented along the embryonic axis, forming clonal strings (doi 10.1242/dev.120.2.265). This is clone-level, tied to the tissue axis.
    - Vertebrate gastrulation (doi 10.1038/nature02796, doi 10.1242/dev.125.6.983, doi 10.1242/dev.127.1.87, doi 10.1016/j.devcel.2010.10.004, doi 10.1002/dvdy.22687, doi 10.1038/ncb2632, doi 10.1038/s41467-020-17196-3, doi 10.1016/j.celrep.2013.08.043) and other systems (doi 10.1242/dev.202862, doi 10.1242/dev.066431, doi 10.1101/2024.06.24.600402) describe orientation as tissue-level polarity, shape or mechanics. None compares mother and daughter axes against a local control.
  - In kind: axis memory across consecutive divisions, cell autonomous beyond the tissue, is reported in fly neural stem cells, and successive-axis patterns are reported in the early worm embryo. A held-out test in gastrulating mouse tissue, with a same-time neighbour control, is not found.

## What is left that is ours
The quantified, held-out test in gastrulating mouse tissue (mother-to-daughter axis similarity beyond same-window neighbours). The phenomenon (axis memory across consecutive divisions, cell autonomous) is published in fly neural stem cells. Held for Bence under RETRO_2026-11-26 decision 1 (in-kind items), as QS2-N1 is.
