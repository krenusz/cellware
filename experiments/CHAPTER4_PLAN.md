# Chapter 4 plan: monoclonal gastruloids (GSE291244, Regalado/Qiu et al., Cell 2026)

Draft 2026-11-07. Nothing analysed yet. Paper read in full (open access; key sections: monoclonal
protocol, sci-RNA-seq3 of 144 gastruloids, intra-gastruloid lineage trees, two-epoch "tree of
trees", oligoclonal founder multiome).

## What the data hold

| experiment | units | per unit | lineage | our status |
|---|---|---|---|---|
| sci-RNA-seq3, 144 monoclonal gastruloids (121 assigned) | gastruloid | ~1,700 cells, 16 cell types | DNA Typewriter tree inside each gastruloid | cell-to-gastruloid step must be rebuilt from tape files |
| two-epoch, 106 gastruloids from one founder of founders | gastruloid, founders related by a tree | cell types (10x) | tree across founders and within gastruloids | rebuild needed (`loci*.csv` 374 MB, Debris-seq tapes) |
| oligoclonal founders: 107 mono + 32 poly gastruloids, mESC multiome | founder clone cluster (28; 26 with >= 200 cells) | gastruloid cell types + mESC RNA/ATAC | clone barcodes only | **assignment public; partitioned (train 9 / A 5 / B 9 / C 5)** |

## Questions, in order of power

1. **Memory per division (our #25), two-epoch data.** How fast does composition similarity decay with
   founder lineage distance? Units: ~100 gastruloids, partition by first-epoch clade. Best-powered
   question; needs the tree rebuild first. This is where chapter 3's open memory question can be
   settled with lineage instead of passages.
2. **Rescue = clonal dominance? (our #22), founder data.** The paper finds polyclonal gastruloids are
   82% one founder (clone 29) and less diverse: "buffering" looks like takeover. Test: does a
   founder's growth in monoclonal gastruloids predict its share in polyclonal ones? Few units (26),
   so a train fit plus one look at most; check H14 power first.
3. **Founder state predicts fate (the paper's Fig. 5).** Already published (Tet1, Dlk1-Dio3); with 9
   train clusters our own model selection is weak. Only as a check that our state model reproduces it.

## First steps

1. Read the authors' Section 1-3 scripts (text only) and list the exact tape files needed.
2. Rebuild cell-to-gastruloid for the two-epoch set; partition by first-epoch clade before any statistic.
3. Pre-register Q1 with a power block (H14) from simulated decay curves.
