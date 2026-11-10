# Chapter 6 credit ledger

This file answers Bence's question "how much credit is that?" for every chapter 6 result, so each score can be
traced. It holds judgement, not statistics, and is updated after each result.

## Scale (0-10): how far the chapter question is answered

The question: can cells carrying the core model's clone state, with local rules only, build a gastruloid's A-P
domains and elongation, **and predict an unseen readout?**

| points | component | what earns it |
|---|---|---|
| 2 | **Form** | one attached body, one posterior pole, realistic elongation (synthetic) |
| 2 | **Clone order** | clone placement predicted with no clone-fit parameter (Q1) |
| 2 | **Mechanism** | the clone state, not the mix ratio or force strength, sets the boundary (Q2) and elongation (Q3) |
| 4 | **Unseen prediction** | a powered, robust prediction confirmed once on sealed data (S2, H13 last partition) |

**Discounts:**
- results found after several model versions;
- targets already seen;
- results exactly at a threshold;
- borrowed mechanisms.

## Ledger

| date | result | Form | Clone order | Mechanism | Unseen | total | why |
|---|---|---|---|---|---|---|---|
| 2026-11-09 | C6-S7..S7g: no version passes step 0 | 0 | 0 | 0 | 0 | **0** | adhesion alone gives 2-3 poles or a broken body |
| 2026-11-09 | **C6-S7h step 0 passes** (polar 0.88, 1 pole, attached, elongation 0.50) | 1.5 | 0 | 0 | 0 | **~3 with the chapter 1-3 base** | first full pass; discounted because 8 versions were tried and the pull is borrowed (Guan) |
| 2026-11-09 | shape benchmark (AR 2.0 vs real 1.2-2.2) + ingredient table | 2 | 0 | 0 | 0 | **~3.5** | realistic shape, but the real range is wide, rests on 1 paper read by eye, and the table describes the model, not biology |
| 2026-11-10 | **C6-S7h clone checks:** Q1 rho exactly 0.90, Q2 fail, Q3 fail, S2 sign flips | 2 | 0.5 | -0.5 | 0 | **~3** | Q1 passes at the bar, but the targets were seen and the order is partly carried in by the clone states; Q2/Q3 show the mix ratio and crawling strength, not the clone state, set the outcome |
| 2026-11-10 | **C6-S7j** 30 repeats: step 0 holds on average (polar 0.77, 1/3 of single runs lack a pole); **elongation before pole (8/10), opposite to biology**. **C6-S7i:** Q2/Q3 failures come from crawling (axis misaligned with pole; saturation), not from the clone-state or identity rule | 1.5 | 0.5 | -0.5 | 0 | **~2.5** | form downgraded (fragile pole, wrong order of events); the diagnosis is understanding, not evidence, but it locates one fixable cause |
| 2026-11-10 | **C6-S7k** posterior-only crawling after onset: step 0 FAIL (no elongation, 0.16-0.28 vs 0.20); pole robust at 200k MCS (polar 1.1-1.4, 0% multipolar, every point) | 1.5 | 0.5 | -0.5 | 0 | **~2.5** | pole now robust, elongation mechanism still missing; no change |
| 2026-11-10 | **C6-S7l** delayed all-cell crawling + alignment: step 0 pass; **Q1 pass** (5/5, rho 0.90, about 1/2 magnitude), **Q2 pass** (slope 0.18 vs data 0.17), **Q3 pass** (monotone); S2 power 0.16 / 0.54, not powered | 2 | 0.75 | 1 | 0 | **~3.75** | first model passing all open checks with the clone rule unchanged and the cause found by diagnosis; capped because Q2/Q3 targets were seen, 12 versions (H18), Q1 at the bar, AR 3.8 high; no unseen test exists |

The "3/10" includes about 1 point carried over from chapters 1-3 (G1a: clone propensity replicated on held-out
data, 7/7). Chapter 6 itself has earned about 2.

## What would move it

| result | expected change |
|---|---|
| C6-S7i diagnosis (done) | 0: located the cause (crawling) |
| a fix, pre-registered, that re-passes step 0 + Q1 and newly passes Q2/Q3 | +1 to +1.5 (Q2/Q3 targets were seen, so capped) |
| S2 powered **and** robust to the state assumption, then a sealed pass | +3 to +4 |
| sealed fail with power >= 0.8 | chapter answered "no"; stop |
