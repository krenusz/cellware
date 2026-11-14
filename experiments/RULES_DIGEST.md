Cellware rules digest (full text: experiments/program.md; this digest is injected every turn).
Hold-out
- H1 Seal a data set before its first statistic. H2 Held-out data is read only through the oracle.
- H3 A sealed file's name appears only in its listed reader. H4 A sealed file's sha256 must be unchanged.
- H5 Check provenance before calling a file untouched. H6 One look; code and predictions committed first; report as it comes.
- H7 Spent is spent. H8 "moderate"/"strong" only for oracle-confirmed results.
- H9 Format checks show columns, tags and counts, never outcomes. H10 Unsure whether data is held out: ask Bence.
- H11 Train 40% + sealed A/B/C by unit hash; one look spends one partition. H12 Train cross-validation picks the model (selection/<exp>.json).
- H13 The last partition needs Bence's verbatim OK.
Power (one rule, three gates; arithmetic only from experiments/power.py)
- H14 Oracle: power >= 0.8 from structural counts, or no look. H17 Selection: power from the model's own simulated statistic.
- H19 Chapter gate: data in hand; primary test power >= 0.8 from the subset's own sealed counts; test within 3 steps.
- Effect and sd come from named used results; list every spent result that constrains an input.
- H22 Name the treatment-assignment unit at S0; < 3 per arm for a between-condition contrast = stop. Re-check on every amendment.
Before a pre-registration
- H15 Pre-flight block: unit, readout, used data first, power, exposure, literature access, readout gate.
- H16 Serves: line naming the current FOCUS step; no work outside the chapter question without Bence's OK.
- H21 Readout gate: outcome distribution shown; >= 30 units per side; statistic differs between models; units that carry it stated.
- Practice: line (program.md checklist: synthetic smoke tests, cold start, analytic regime, measurement physics, adequacy gate, tick-list).
Chapters and retros
- H18 Retro before a 5th amendment within one step. H20 Before any code: source methods read, top 5 papers read, minimal published mechanism, value stated; stop reading at sealed outcomes.
- Every retro lesson lands as a rule, a practice line, a memory item or "not adopted".
Result reports
- Self-challenge before the conclusion: suspicious? other explanations? how many units? used data enough? already published?
- No outreach. Authors' result tables for sealed sets stay unread. FOCUS.json changes need a commit quoting Bence.
