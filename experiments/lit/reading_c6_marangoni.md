# Reading notes: Marangoni-like tissue flows in gastruloids (C6-S7, 2026-11-10)

Bence: "go 2 3" (2 = read the Marangoni preprint, flagged by RETRO_2026-11-10). Paraphrased.

**Sources:**
- Gsell, Tlili, Merkel, Lenne, bioRxiv 10.1101/2023.09.22.559003 (v1 2023, v2 2024; published as Nat Phys 2025,
  10.1038/s41567-025-02802-2).
- The first part of v1 and v2 was read as page text [V]. The rest came only as a tool summary [S]; numbers from [S]
  are second-hand.
- The paywalled companion (Nat Phys 2025, 10.1038/s41567-025-02840-w, "Marangoni flows underlie symmetry breaking
  in early mouse gastruloids") was **not read**.

## What they show

- **System:** mouse gastruloids, standard protocol (CHIR 48-72 h) [V], 100-200 cells [S]. Live two-photon imaging
  at 78-88 h AAF, 9 gastruloids for the flow analysis [V]. Reporters T/Bra-GFP and E-cad-GFP.
- **Flow:**
  - A large-scale **recirculating tissue flow**; the central flow points toward the high-T/Bra side [V].
  - About half the flow sits in 5 low-order modes [V].
  - The recirculation direction lines up with the T/Bra dipole [V].
- **Mechanism (model):**
  - The A-P state is one field phi (-1 anterior, +1 posterior). Fate-dependent active stress gives an interface
    tension between domains.
  - The surface tension also depends on fate, and the **anterior surface is stiffer**, linked to E-cadherin [S].
  - The surface-tension difference drives a Marangoni flow (overdamped Stokes, lattice Boltzmann [S]).
- **Contribution to the pole:** advection by the flow gives **about 1/3 of the polarisation**; reaction-diffusion
  of the signalling field gives the rest [V, Fig 2f]. They leave open whether the flow drives the pole or the
  reverse.
- **Not tested:** the molecular cause of the tension difference (adhesion, contractility or motility; knockouts
  left for future work), and anything about mixtures, clones, or cells unable to drive the flow [S].
- **Fits:**
  - tension ratio about 0.8-2.5, surface-tension difference -0.2 to -1 [FL Fig 4g, 5];
  - fusion of two aggregates (++ vs +-) reproduced with and without a tension difference [S].
  - No flow speed in micrometres per hour in the text read.

## What it means for chapter 6

1. **It is a physical candidate for our long-range pull.** Guan's 1/d^2 pull is phenomenological. Marangoni
   flow is a tissue-scale advection that brings posterior cells together, and it comes from a fate-dependent
   surface-tension difference.
2. **The direction matches one of our energy tables.**
   - Anterior surface stiffer means J_Am > J_Pm. That is our **W4** (J_Am 25, J_Pm 20), not the W1 we chose
     (symmetric).
   - In C6-S7h, W4 + pull elongated **without** crawling (0.45). This fits a stress-driven tissue flow that both
     gathers and stretches.
   - It is not evidence yet: we never ran W4 through the data checks.
3. **A Potts (CPM) model cannot produce a true Marangoni flow.** Metropolis dynamics have no momentum and no
   Stokes flow. A faithful version would need either:
   - an advection term (a velocity field from the surface-tension gradient applied to cell centroids, like our
     force term but derived from tension, not from 1/d^2); or
   - a continuum model.
4. **Only about 1/3 of polarisation comes from flow** in their analysis. Signalling (reaction-diffusion) does the
   rest. Our model has no signalling field since C6-S7 (identity from clone state only). That is a known gap, not
   a contradiction: the C6-S3 wave-pinning field was the signalling route, and it traded off against clone
   placement.
5. **No new unseen prediction for our clone data.**
   - The paper says nothing about mixing, clones or flow-defective cells.
   - Their fusion experiment (++ vs +- aggregates, fusion angles) is a literature readout, not clone data.
   - It could be a weak out-of-sample check for a model with flow. Our pull model has not been asked to predict
     it.

## Verdict

- **It supports keeping a long-range gathering term.** It suggests the biologically better form is a
  surface-tension asymmetry (anterior stiffer) driving tissue flow, rather than a 1/d^2 attraction.
- **It does not give the powered unseen test chapter 6 needs.** Porting it would be a new model (a 14th version;
  H18 retro due) with no new data to test against.
- **Worth reading next:** the paywalled companion (Nat Phys 2025, Springer Nature). Its text may hold the
  perturbation data (E-cad, size) missing here. Access via a library only; no contact with anyone.
