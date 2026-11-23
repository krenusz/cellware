"""CAND-F-P1b phase B: clones per mouse in GSE299003 / GSE307390 vs the source's 7 published counts.

Reads only the clone, mouse and cell-type columns named in EXPERIMENT_LOG (CAND-F-P1b amendment 1). The cell-type
column is reduced to one flag (level name contains 'HSC'). No clone id, other cell-type level, bias column or
expression value is written.
"""

import collections
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from candf_p1b_struct import FILES, ensure_r, fetch  # noqa: E402

OUT = os.path.join(HERE, "results_cand", "candf_p1b_counts.json")
PUBLISHED = sorted([14, 22, 24, 27, 33, 64, 86])
CFG = {
    "GSE299003": {"clone": ["larry", "larry_gfp", "larry_gfp_filt", "larry_unique"],
                  "mouse": ["Mouse", "sample_mouse"], "hsc": ["CellType", "MainLineages"]},
    "GSE307390": {"clone": ["larry_merged", "larry_gfp", "larry_gfp_sample", "larry_sapphire"],
                  "mouse": ["sample"], "hsc": ["lineages", "main_lineages"]},
}
R_COUNTS = r"""
args <- commandArgs(TRUE)
suppressMessages({library(SeuratObject); library(jsonlite)})
cfg <- fromJSON(args[2])
m <- readRDS(args[1])@meta.data
PH <- "^(na|none|nan|unassigned|negative|no[._ -]?(larry|barcode|bc)|not[._ -]?detected|0|)$"
out <- list(); shapes <- list(); hsc_levels <- list()
for (cc in cfg$clone) {
  v <- as.character(m[[cc]])
  sh <- gsub("[0-9]+", "N", gsub("[ACGT]{4,}", "S", v))
  shapes[[cc]] <- as.list(head(sort(table(sh, useNA = "ifany"), decreasing = TRUE), 6))
  names(shapes[[cc]]) <- ifelse(is.na(names(shapes[[cc]])), "<NA>", substr(names(shapes[[cc]]), 1, 24))
  ok <- !is.na(v) & !grepl(PH, trimws(v), ignore.case = TRUE)
  for (mc in cfg$mouse) {
    u <- as.character(m[[mc]])
    keep <- !grepl("Unassigned", u)
    for (hc in cfg$hsc) {
      lv <- as.character(m[[hc]]); isH <- grepl("HSC", lv)
      hsc_levels[[hc]] <- sort(unique(lv[isH]))
      r <- list()
      for (k in sort(unique(u[keep]))) {
        i <- which(u == k & ok)
        a <- table(v[i]); h <- unique(v[i][isH[i]])
        r[[k]] <- list(cells = length(i), any = length(a), ge2 = sum(a >= 2),
                       ge2_hsc1 = sum(a >= 2 & names(a) %in% h))
      }
      out[[paste(cc, mc, hc, sep = "|")]] <- r
    }
  }
}
write_json(list(counts = out, shapes = shapes, hsc_levels = hsc_levels), args[3], auto_unbox = TRUE)
"""


def matched(vals):
    have = collections.Counter(vals)
    return sum(min(have[x], c) for x, c in collections.Counter(PUBLISHED).items())


def main():
    os.environ.setdefault("R_LIBS_USER", os.path.expanduser("~/Rlib"))
    ensure_r()
    res, best = {}, []
    with tempfile.TemporaryDirectory() as d:
        script = os.path.join(d, "counts.R")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(R_COUNTS)
        for name, url in FILES.items():
            rds, cf, js = (os.path.join(d, name + x) for x in (".rds", ".cfg.json", ".json"))
            sha = fetch(url, rds)
            with open(cf, "w", encoding="utf-8") as fh:
                json.dump(CFG[name], fh)
            subprocess.run(["Rscript", script, rds, cf, js], check=True)
            with open(js, encoding="utf-8") as fh:
                r = json.load(fh)
            r["sha256"] = sha
            res[name] = r
            os.remove(rds)
            for key, mice in r["counts"].items():
                for variant in ("any", "ge2", "ge2_hsc1"):
                    vals = [m[variant] for m in mice.values()]
                    best.append({"series": name, "columns": key, "variant": variant, "mice": len(vals),
                                 "published_matched": matched(vals)})
    best.sort(key=lambda b: (-b["published_matched"], b["series"], b["columns"], b["variant"]))
    top = best[0]["published_matched"]
    verdict = "mapped" if top == len(PUBLISHED) else "units unresolved"
    out = {"published": PUBLISHED, "verdict": verdict, "best": best[:12], "series": res}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps({"verdict": verdict, "best": best[:3]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
