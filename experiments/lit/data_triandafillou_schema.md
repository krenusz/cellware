# Triandafillou 2026 atlas: schema read for C6-S8d (2026-11-10)

Bence: "go both". Source: Zenodo 10.5281/zenodo.22233771 (CC BY 4.0). Untrusted data, in
`data/ext_triandafillou/`. Locked by sha256 in `SHA256SUMS` and `members/SHA256SUMS` before any statistic.
**Read so far:** file names, CSV headers, row counts, JSON key structure and tag vocabularies. **No value read.**
Notebooks in plotScripts.zip (their outputs show results) are **not opened**.

| set | what | n | readable for us |
|---|---|---|---|
| `bf-images/quant/*.csv`+`.json` | brightfield outlines (polygons) + blob metrics (Elongation, Eccentricity, Circularity, Solidity, Area, ...) | 6 plates, **all at 120 h**: 300 cells x 2 plates (2024-01-19 n 88, 2024-09-01 n 86); 100 cells x 4 plates (2025-04-04, p2/p3, A/B: 93, 88, 84, 90) = **529** | outlines give our own `moment_elong` / 1 - circularity with no time calibration |
| `sg/<date>/roi*/areas.json` | per seqFISH section: outline (`shape`), hand-drawn `AP axis` line, `A_area` / `P_area` polygons | about 26 sections | **scale-free shape-pole test**: angle between the shape's long axis and the AP axis; P area share; single vs multiple P areas |
| `sg/.../annotations.json`, `decode.csv` | nuclei polygons, decoded transcripts | about 26 | cell types would need the authors' typing; not needed for the tests above |
| `misc/*.csv` | summaries (transcripts per nucleus, % correct by date, NN distances) | small | not needed |

**Correction to the earlier note:** "3 days" are three imaging dates, not a time course. Every brightfield
gastruloid is at 120 h, so no model-time-to-hours calibration is needed.

## Candidate unseen tests (to pre-register; values unread)

1. **Axis alignment (seqFISH sections, scale-free).** The model predicts the distribution of |cos| between the
   long axis and the P-A direction (C6-S7i/l `align`). The data has the authors' AP line and outline per section.
   n about 26 is small. The prediction comes from arm a (and arm c as a robustness check), with power under H17.
2. **Seeding-size effect (brightfield, 100 vs 300 cells).** The model predicts the direction and size of the
   difference in elongation when the starting cell number is cut threefold. This needs an N-scaled run (a
   disclosed assumption: model N is proportional to seeded cells). n 355 vs 174, so power is high. Note:
   size-to-elongation trends appear in the literature (Fiuza 2024 size series, already read), so the direction
   is **partly exposed**; only the magnitude would be unseen.
3. **Spread of elongation at 120 h** (plate-to-plate and within plate). Not scale-free; it needs the C6-S7l
   conversion (0.180 / model gap), which came from a different line. Weakest.

Neither test touches the clone claim. Under the CREDIT_C6 scale they count only for Form/Unseen, not Clone order.
