# Reading notes: do posterior (T/Bra, Wnt-high) cells start on the outside? (C6-S7e, 2026-11-09)

Bence: "okey do more literatur research for that, otherwise we save it as worst case model". Written and committed
**before** the C6-S7e results were looked at. Paraphrased; [S] = seen only in a tool summary, not verbatim.

| source | system, protocol | where T/Bra+ (posterior-type) cells are before the pole | how the pole forms |
|---|---|---|---|
| Hashmi et al. 2022 eLife 11:e59371 | mouse gastruloid, **standard** (CHIR 48-72 h), free-floating | **72 h: T-Bra+ cells at the periphery** (Fig 3C radial profile, n 7), wrapped round an E-cad+ core; T-Bra-only cells "sparsely wrapped" (patchy: 13 pieces for 31 cells, Fig 3B). T-Bra pole "emerges from a spatially homogeneous distribution at day 3" (Fig 1C-D) | tissue-scale flow toward the future tip (Fig 4, optical flow [S]); pole by 96 h |
| Suppinger et al. 2023 Cell Stem Cell 30:867 | mouse gastruloid, **standard** | Sox2 gradient high in the core at 36 h; CHIR makes it binary: **Bra only in the former Sox2-low (outer) regions**, Sox2+ core (Fig 3C); Bra at 60 h "salt-and-pepper yet peripheral" | opposite-pole bias from 72 h; Sox2 core moves to the anterior tip (mechanism not stated) |
| Serrano Najera, Delahaye, Steventon 2025 bioRxiv 2025.08.04.668470 | mouse gastruloid, standard; free-floating vs laminin | free-floating: pole from a salt-and-pepper start (citing van den Brink 2014, Hashmi 2022). On laminin: **T/Bra ring at the edge on day 3**, cells "trapped at their sites of induction around the gastruloid periphery" | free-floating: one pole by rearrangement; laminin: several poles (cells cannot move) |
| Sagy et al. 2019 Development 146:dev181917 | mouse embryoid bodies (no CHIR by default) | Bra onset **in the outer shell**, usually one locus at the contact point; isotropic peripheral onset with CHIR or in Matrigel (Fig 1-2) | expression polarises; sorting not shown [S] |
| Oriola et al. 2024 bioRxiv 2024.12.16.628776 v2 | mouse gastruloid **without** CHIR, sorted T+/T- mixes | **opposite:** 24-48 h T+ cells in the core, T- (pluripotent or later mesoderm) at the periphery (Fig 2D, 3C) | T+ core then polarises through an outer mesoderm crust (model) |
| McNamara et al. 2024 Nat Cell Biol 26:1832 | mouse gastruloid, standard | 96 h: Wnt-high patches with no axial bias (radial position not reported) | motile Wnt-high cells cluster and merge; sorting by adhesion |

## Verdict

- **In the standard protocol (CHIR 48-72 h, which is the protocol of our clone data), posterior-type cells do
  start mostly on the outside before the pole:** Hashmi (quantified, n 7) and Suppinger agree. Sagy agrees in
  embryoid bodies. My earlier "weak biological support" was wrong; it came from reading McNamara's "patches
  without axial bias" as "scattered", when those patches can sit on the surface.
- **Caveats that stay:**
  1. The outer layer is **patchy, not a uniform shell** (Hashmi: 13 separate pieces; Suppinger:
     salt-and-pepper yet peripheral). Our outer start puts every P cell in a full ring, which is easier than reality.
  2. **Timing:** the peripheral layer exists by ~60-72 h; we assign identity and position at the start, so
     the model skips how the shell arises (Suppinger: Sox2 core gradient from 36 h; Oriola: depends on
     protocol, reversed without CHIR).
  3. Without CHIR the arrangement reverses (Oriola), so the outer start is protocol-specific.
- **Consequence for C6-S7e:** the outer-start arm is a legitimate model of the standard protocol, not a
  worst-case model. A pass in it can be claimed for "from the 72 h peripheral state onward", with caveats 1-2.
  A stricter version that matches the data better is **patchy outer start** (P cells on the outer rim but in
  separate clusters, not a full ring). That would need its own pre-registration and Bence's OK.
