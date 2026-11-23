"""CAND-F-P1b phase A: schema of the authors' processed objects in GSE299003 and GSE307390 (no statistic).

Reads only: object class, dimensions, metadata column names, each column's type and number of distinct values, and
the level names of columns whose NAME looks like a mouse/sample/hashtag id. No cell-type level, clone id, per-cell
value or expression value is written. Phase B (per-mouse clone counts) is a separate run once the columns are known.
See EXPERIMENT_LOG CAND-F-P1b.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_cand", "candf_p1b_schema.json")
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series/"
FILES = {
    "GSE299003": GEO + "GSE299nnn/GSE299003/suppl/GSE299003_Figures-Figure3.rds",
    "GSE307390": GEO + "GSE307nnn/GSE307390/suppl/GSE307390_Figures-Figure1.rds",
}
R_SCHEMA = r"""
args <- commandArgs(TRUE)
suppressMessages({library(SeuratObject); library(jsonlite)})
ID <- "mouse|sample|hash|receptor|orig|donor|recipient|batch|timepoint"
meta <- function(m) lapply(setNames(names(m), names(m)), function(k) {
  x <- m[[k]]
  r <- list(type = class(x)[1], n_distinct = length(unique(x)), n_missing = sum(is.na(x)))
  if (grepl(ID, k, ignore.case = TRUE) && r$n_distinct <= 40) r$levels <- sort(as.character(unique(x)))
  r
})
desc <- function(o, depth = 0) {
  r <- list(class = class(o)[1])
  if (inherits(o, "Seurat")) {
    r$cells <- ncol(o); r$assays <- names(o@assays); r$reductions <- names(o@reductions); r$meta <- meta(o@meta.data)
  } else if (is.data.frame(o)) {
    r$rows <- nrow(o); r$meta <- meta(o)
  } else if (is.list(o) && depth < 2) {
    r$items <- lapply(o, desc, depth = depth + 1)
  }
  r
}
write_json(desc(readRDS(args[1])), args[2], auto_unbox = TRUE, pretty = TRUE)
"""


def sh(cmd):
    subprocess.run(cmd, check=True)


def ensure_r():
    if not shutil.which("Rscript"):
        sh(["sudo", "apt-get", "update", "-qq"])
        sh(["sudo", "apt-get", "install", "-y", "-qq", "r-base"])
    code = (
        'options(repos="https://packagemanager.posit.co/cran/__linux__/noble/latest",'
        'HTTPUserAgent=sprintf("R/%s R (%s)",getRversion(),paste(getRversion(),R.version$platform,'
        'R.version$arch,R.version$os)));'
        'dir.create(Sys.getenv("R_LIBS_USER"),recursive=TRUE,showWarnings=FALSE);'
        '.libPaths(Sys.getenv("R_LIBS_USER"));'
        'install.packages(c("SeuratObject","jsonlite"),lib=Sys.getenv("R_LIBS_USER"))'
    )
    sh(["Rscript", "-e", code])


def fetch(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": "cellware-seal/1.0"})
    for i in range(8):
        try:
            h = hashlib.sha256()
            with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as fh:
                while b := r.read(1 << 20):
                    h.update(b)
                    fh.write(b)
            return h.hexdigest()
        except Exception as e:  # noqa: BLE001
            print(f"retry {i + 1} {url}: {e}", flush=True)
            time.sleep(10 * (i + 1))
    raise RuntimeError(url)


def main():
    os.environ.setdefault("R_LIBS_USER", os.path.expanduser("~/Rlib"))
    ensure_r()
    res = {}
    with tempfile.TemporaryDirectory() as d:
        script = os.path.join(d, "schema.R")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(R_SCHEMA)
        for name, url in FILES.items():
            rds, js = os.path.join(d, name + ".rds"), os.path.join(d, name + ".json")
            sha = fetch(url, rds)
            sh(["Rscript", script, rds, js])
            with open(js, encoding="utf-8") as fh:
                res[name] = {"file": url.rsplit("/", 1)[-1], "sha256": sha, "schema": json.load(fh)}
            os.remove(rds)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print("schema written", {k: v["schema"].get("class") for k, v in res.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
