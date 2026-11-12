"""Discovery labeller, merge stage. Reads the metadata crawl (raj_tree.jsonl) and the group rules that the
labelling agents wrote (rules/*.json), and writes one label row per file to raj_labels.tsv.
Metadata only: opens no data file.

Rule format (list of objects): {"glob": "raw_data/SG/**", "usefulness": "B", "questions": "Q4",
"units": "...", "basis": "metadata", "note": "...", "seal": optional, "kind": optional}.
Rules are applied in order (file order, then list order); a later match overrides an earlier one."""
import collections, fnmatch, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "raj_tree.jsonl")

KIND = {
    ".fastq": "raw_seq", ".fq": "raw_seq", ".gz": None, ".bam": "alignment", ".bai": "alignment", ".bw": "peaks",
    ".narrowpeak": "peaks", ".bed": "peaks", ".xls": "peaks", ".csv": "table", ".tsv": "table", ".txt": "table",
    ".xlsx": "table", ".h5": "counts", ".h5ad": "counts", ".hdf5": "counts", ".json": "annotation",
    ".tif": "image", ".tiff": "image", ".nd2": "image", ".png": "image", ".jpg": "image", ".svg": "figure",
    ".pdf": "figure", ".ai": "figure", ".mp4": "movie", ".avi": "movie", ".mov": "movie", ".r": "script",
    ".py": "script", ".ipynb": "script", ".rmd": "script", ".m": "script", ".sh": "script", ".rds": "object",
    ".html": "motif", ".motif": "motif", ".motifs": "motif", ".log": "other", ".md": "other",
}


def kind_of(path):
    p = path.lower()
    if "homer" in p:
        return "motif"
    base = p[:-3] if p.endswith(".gz") else p
    ext = os.path.splitext(base)[1]
    return KIND.get(ext) or "other"


def load_rules():
    rules = []
    for f in sorted(glob.glob(os.path.join(HERE, "rules", "*.json"))):
        for r in json.load(open(f, encoding="utf-8")):
            r["_src"] = os.path.basename(f)
            rules.append(r)
    # Broad defaults (glob with at most one "/") go first, so an area's catch-all never overrides
    # another area's specific rules. Stable sort keeps file and list order otherwise.
    return sorted(rules, key=lambda r: 0 if r["glob"].count("/") <= 1 else 1)


def main():
    files = [json.loads(l) for l in open(TREE, encoding="utf-8")]
    files = [f for f in files if not f["is_dir"]]
    rules = load_rules()
    rows, unmatched = [], 0
    for f in files:
        row = {"id": f["id"], "path": f["path"], "bytes": f["bytes"] or 0, "kind": kind_of(f["path"]),
               "seal": "new", "local": "no", "usefulness": "?", "questions": "-", "units": "", "basis": "metadata",
               "note": "", "rule": ""}
        hit = False
        for r in rules:
            if fnmatch.fnmatch(f["path"], r["glob"]):
                hit = True
                for k in ("usefulness", "questions", "units", "basis", "note", "seal", "kind", "local"):
                    if k in r:
                        row[k] = r[k]
                row["rule"] = f'{r["_src"]}:{r["glob"]}'
        unmatched += not hit
        rows.append(row)
    cols = ["id", "path", "bytes", "kind", "seal", "local", "usefulness", "questions", "units", "basis", "note", "rule"]
    with open(os.path.join(HERE, "raj_labels.tsv"), "w", encoding="utf-8", newline="\n") as o:
        o.write("\t".join(cols) + "\n")
        for r in rows:
            o.write("\t".join(str(r[c]).replace("\t", " ").replace("\n", " ") for c in cols) + "\n")
    by = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        by[r["usefulness"]][0] += 1
        by[r["usefulness"]][1] += int(r["bytes"])
    print("files", len(rows), "unmatched", unmatched)
    for k in sorted(by):
        print(f"  {k}: {by[k][0]} files, {by[k][1] / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
