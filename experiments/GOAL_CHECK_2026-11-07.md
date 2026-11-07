# Firm check and goal alignment, 2026-11-07

Asked by Bence: "now you are drifting! ... we have strict rule to follow, biology is really wide
topic, do a firm check and goal alignment". Measured against program.md, not against what seemed
interesting.

## 1. What program.md says we are doing

- **Long term:** model stem-cell fate, then morphogenesis (cells with local rules build a patterned tissue).
- **Plan toward space (2026-11-06):** the clone-level **core model** (`core_model.py`: clone state,
  instruction shift, selection on state, memory rho) becomes the **cell-level prior of the tissue
  simulator**. Chapter 3 test: *"The model must reproduce clone spatial propensities and the
  pure-vs-mixed elongation difference from clone states plus local rules."*
- **Chapter 4 (Bence, 2026-11-06):** the **mechanism of mixing rescue / division of labour**
  (#22, #24), with data that has per-clone composition of pure and mixed gastruloids.
  "Not started; chapter 3 first finishes."
- **Stopping rule:** a misfit earns new work only if fixing it could move the chapter's answer;
  otherwise record it and move on.

## 2. Where the work went, experiment by experiment

| exp | serves the chapter-3 test? | verdict |
|---|---|---|
| G1-G4 | yes: propensity, elongation, fade, rescue mechanisms | on goal |
| G5, G7 | yes: simulator + sealed test | on goal, **but built outside `core_model.py`** |
| G6 | yes: sealed confirmation | on goal |
| G8 | memory rate model for one sub-rule | marginal: could not move the chapter answer (stopping rule) |
| G9-G12 | drug / Wnt compression, "series gain" | **drift**: not in the chapter test; chased because the data was there |
| G13 | DiffAssay growth bias | **drift** (cheap, but off-question) |
| Chapter 4 work | founder memory per division, founder state, DiffAssay "domain size" | **drift**: I replaced Bence's chapter-4 question (rescue mechanism) with questions the GSE291244 data happened to allow |

## 3. The three real gaps

1. **The core model was never carried into space.** `g5_gastruloid.py` / `g7_model.py` import no
   part of `core_model.py`; the gastruloid clone "state" is a fitted polarity number, not the core's
   clone state z with memory rho. The project's central plan (one model across chapters 1-3) is
   therefore not yet tested. This is the biggest drift and it happened quietly.
2. **Elongation rescue is still a descriptive rule** in the simulator; the chapter-3 test asked for
   it "from clone states plus local rules". Open.
3. **No scope guard.** H1-H15 protect sealed data and test design; nothing stops a well-run
   experiment on the wrong question.

## 4. Fixes

- **Scope guard (program.md, enforced with H15):** every pre-registration carries a `Serves:` line
  naming the goal element and the chapter question, and a stopping-rule sentence (how the result
  could change the chapter answer). The oracle refuses a look without it. New questions outside the
  current chapter need Bence's OK in chat, quoted.
- **Parked (recorded, no further work):** series gain / drug compression (#27), DiffAssay (G13),
  founder memory per division and founder state (GSE291244 sections 2-4), #28 dominance.
- **Stale text:** program.md "Fixed setup" still describes the 2023-24 PPO tissue runs; it does not
  apply to chapters 1-3 and is marked as such.

## 5. Aligned next steps (in order; nothing else)

1. **Close chapter 3 by its own definition:** rebuild the gastruloid simulator on `core_model.py`
   (clone state z with memory rho from the core; fate/propensity via the core's state-to-logit map;
   rank sorting + proportion feedback as the local rules), then check it still reproduces #20,
   #23 and #26 on open and spent data. No sealed look unless a model passes H12 + H14 + H15.
2. **Then the chapter-3 open item:** elongation rescue from local rules (candidate mechanisms from
   the literature map: cross-lineage dependence of mesoderm and spinal-cord branches, Dev Cell 2025;
   cell-non-autonomous compensation, chimeric gastruloids 2022), tested on G2/G4 open data.
3. **Chapter 4 (rescue mechanism) only after 1-2,** with a data search restricted to its stated
   requirement (per-clone composition of pure and mixed gastruloids) and the pre-flight run on each
   candidate before any download.
