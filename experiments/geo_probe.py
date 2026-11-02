"""Probe Camellia-seq GEO data on GitHub Actions (NCBI blocks the local machine): sample titles /
characteristics / file names for the RNA, methylation and lineage series, plus the first lines
of one file of each kind. Writes experiments/results_track_b/geo_probe.json.

    python experiments/geo_probe.py
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
import time
import urllib.request

HERE = os.path.dirname(__file__)
SERIES = ("GSE222246", "GSE222328", "GSE222479")


def get(url, binary=False, tries=6):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            return data if binary else data.decode("utf-8", "replace")
        except Exception:  # noqa: BLE001  NCBI throttles: back off and retry
            time.sleep(5 * (i + 1))
    raise RuntimeError(url)


def samples(acc):
    text = get(
        f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}&targ=gsm&form=text&view=brief"
    )
    out = []
    for block in text.split("^SAMPLE = ")[1:]:
        gsm = block.split("\n", 1)[0].strip()
        title = re.search(r"!Sample_title = (.*)", block)
        files = re.findall(r"!Sample_supplementary_file_\d+ = (\S+)", block)
        chars = re.findall(r"!Sample_characteristics_ch1 = (.*)", block)
        out.append(
            {"gsm": gsm, "title": title.group(1) if title else "", "files": files, "chars": chars}
        )
    return out


def head(url, n=8):
    url = url.replace("ftp://", "https://")
    raw = get(url, binary=True)
    lines = (
        gzip.GzipFile(fileobj=io.BytesIO(raw)).read(200000).decode("utf-8", "replace").splitlines()
    )
    return {"bytes": len(raw), "head": lines[:n]}


def main():
    out = {}
    for acc in SERIES:
        s = samples(acc)
        kinds = {}
        for smp in s:
            for f in smp["files"]:
                kinds.setdefault(
                    re.sub(r"GSM\d+_", "", f.rsplit("/", 1)[-1]).split("_")[-1], []
                ).append(f)
        out[acc] = {
            "n": len(s),
            "first": s[:3],
            "last": s[-2:],
            "file_kinds": {k: len(v) for k, v in kinds.items()},
            "heads": {},
        }
        for k, v in kinds.items():
            try:
                out[acc]["heads"][k] = head(v[0])
            except Exception as e:  # noqa: BLE001
                out[acc]["heads"][k] = str(e)
    os.makedirs(os.path.join(HERE, "results_track_b"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results_track_b", "geo_probe.json"), "w"), indent=1)
    print(json.dumps(out, indent=1)[:5000])


if __name__ == "__main__":
    main()
