# C10-S0: lineage-label access and extraction price (2026-11-13)

Bence: "Price the extraction". Note only: nothing was downloaded or run, and no outcome was tabulated.

## Measured
- **SRA:** 64 runs (16 samples x 4 lanes of one 10x library each), about 564 GB.
  - No BAMs; the originals were FASTQ (R1, R2, I1, I2).
  - The SRA-normalised file (about 7.6 GB per run) is reachable anonymously on the public AWS bucket.
  - The original FASTQ.gz files were not found at the guessed public bucket names.
- **Four extra runs (GSM7990953-56)** are a separate drug-perturbation experiment (gene expression plus Cell
  Multiplexing libraries), not label amplicons.
- **Label design:** from the authors' `umitools_cross_sample.py`, the construct is
  `...AAGCTTAATTAAGAATTCA NNNNTGNNNNACNNNNGANNNNGTNNNN CTAGGGCCTAGAGGGC...`; the authors use the match regex
  `CA.{4}TG.{4}AC.{4}GA.{4}GT.{4}CT`, then UMI-tools `directional` correction across samples. Their reads came from the
  BAM region `pBA439_UMI_20` (custom reference gene).
- **Column meanings, from the authors' Preprocess Rmd (code only, values never tabulated):**
  - `sisters`: lineage ID shared by pre-treatment sisters (perfect label match); NA without a sister.
  - `drugSens`: pre-resistant / resistant / pre-sensitive (sister recovered or not after treatment). **Outcome.**
  - `prer_r_group`: **family ID linking pre-resistant cells to their post-treatment resistant sisters**, perfect
    matches only; NA otherwise. **Outcome-linked.**

## Consequence: the extraction is probably not needed
The pre-to-post lineage match is already in the locked processed tables (`prer_r_group`, present in both pre and post
tables). What the processed tables lack is only:
- the raw label sequences (needed only to re-derive matches or rescue partial matches);
- labels for post cells without a perfect pre match (they carry NA).

The partition unit can be built from the locked tables:
- a pre-treatment lineage = its `sisters` ID (or the cell itself if it has no sister);
- post cells join their lineage through `prer_r_group`.

That join reads an outcome-linked column, so the partition script may output only partition membership and
**pre-side** unit counts per partition, never post-side or matched counts.

## Price if the extraction is still wanted (estimate, throughput not timed)
| stage | data | runner time (assumed) | cost |
|---|---|---|---|
| Pilot: 1 pre-treatment run | 7.6 GB stream | 20-40 min | $0 (public repo, standard runner) |
| One lane per sample | 16 runs, about 125 GB | about 1 h wall, 16 parallel | $0 |
| All lanes | 64 runs, about 564 GB | about 2 h wall, 20 parallel | $0 |

**Binding constraints:**
- About 14 GB of free runner disk: fetch the 7.6 GB SRA file, then stream reads out with no FASTQ on disk.
- Technical reads must be included (R1 carries the cell barcode and UMI).
- 6 h per job.

**Output:** per sample, cell barcode, UMI and raw label only. The workflow does no pre/post join and prints no
overlap count. Lanes are merged and labels corrected afterwards, then hashed, locked and partitioned before any match
is counted.

## What labels buy (value side)
They move the unit from library (2 per arm) to lineage. A mean pre-to-post shift stays confounded with library
batch. The instruction contrast must be within a library and across lineages: in the post sample, do resistant
families differ from each other beyond what their pre-sisters' states predict? That needs only the family links
already in `prer_r_group`.

## Next (needs Bence)
1. **H20 top-5 read**, then a partition script on the locked tables (pre-side counts only); new `allowed_scripts`
   entry and a fresh quote.
2. **Extraction pilot:** only if the partition shows that the perfect-match families are too few, and partial
   matches would add units.
3. **Stop rule unchanged:** if the H19 power on the train families is < 0.8, stop.
