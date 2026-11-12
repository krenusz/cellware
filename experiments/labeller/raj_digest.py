"""Folder digest of the crawl for the labelling agents: per leaf folder, file count, size, extension counts and
up to 6 example names. Metadata only."""
import collections, json, os, sys

TREE, OUT = sys.argv[1], sys.argv[2]
agg = collections.OrderedDict()
for l in open(TREE, encoding="utf-8"):
    f = json.loads(l)
    if f["is_dir"]:
        continue
    d = os.path.dirname(f["path"])
    a = agg.setdefault(d, {"n": 0, "bytes": 0, "ext": collections.Counter(), "ex": []})
    a["n"] += 1
    a["bytes"] += f["bytes"] or 0
    n = os.path.basename(f["path"])
    a["ext"][n[:-3].rsplit(".", 1)[-1] if n.endswith(".gz") else n.rsplit(".", 1)[-1]] += 1
    if len(a["ex"]) < 6:
        a["ex"].append(n)
with open(OUT, "w", encoding="utf-8") as o:
    for d in sorted(agg):
        a = agg[d]
        o.write(f"{d}\t{a['n']}\t{a['bytes'] / 1e6:.1f}MB\t{dict(a['ext'].most_common(5))}\t{' | '.join(a['ex'])}\n")
print("folders", len(agg), "files", sum(a["n"] for a in agg.values()),
      "GB", round(sum(a["bytes"] for a in agg.values()) / 1e9, 1))
