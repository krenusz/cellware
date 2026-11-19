# <step> novelty check (H25, <date>)

Copy to `experiments/lit/novelty/<step>.md` before FOCUS.json moves to <step>. The commit hook checks it.

- Searched: <sources (OpenAlex, web, bioRxiv), queries, date; gap map file if run>
- Source paper: <the data set's source paper. If its partitions are sealed: claims level only (abstract, section headings, figure titles, stated conclusions; no numbers, panels, tables, supplements), listed here as accepted exposure (H25, Bence 2026-11-18). Otherwise: what was read.>

## Planned findings
- Finding: <what the step expects to show, in one line>
  - Verdict: <published | partly published | not found>
  - Papers: <closest papers, with DOI, and how close>
  - Classic search: <required unless published: `python -I experiments/novelty_search.py <step> --source <doi> --q "<field's own words>" --q "<our words>"`; results_lit/novelty_<step>.json; top 25 titles screened, top 10 relevant abstracts read, source references top 20 screened; what was found>
- Finding: <...>
  - Verdict: <...>
  - Papers: <...>

## What is left that is ours
<the part no paper covers; if nothing, say so and do not open the step>

<!-- If any Verdict is "published" and the step goes ahead anyway, add:
Published but proceeding: Bence: "<his words>"
-->
