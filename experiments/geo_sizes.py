"""List GEO supplementary file sizes for series/samples (run on GitHub Actions: NCBI blocks the
local machine). Writes experiments/results_track_b/geo_sizes.json.

    python experiments/geo_sizes.py GSE222246 GSE222328 GSE222479
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(__file__)


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def listing(url):
    """Apache/NCBI FTP-over-HTTPS directory listing -> {name: size string}."""
    html = get(url)
    return dict(re.findall(r'href="([^"/?]+)".*?\s(\d[\d.]*[KMG]?)\s*$', html, re.M))


def main():
    out = {}
    for acc in sys.argv[1:]:
        text = get(
            f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}&targ=self&form=text&view=brief"
        )
        samples = re.findall(r"!Series_sample_id = (GSM\d+)", text)
        entry = {"samples": len(samples)}
        series_dir = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{acc[:-3]}nnn/{acc}/suppl/"
        try:
            entry["series_files"] = listing(series_dir)
        except Exception as e:  # noqa: BLE001
            entry["series_files"] = str(e)
        sample_sizes = {}
        for gsm in samples[:3] + samples[-2:]:
            try:
                sample_sizes[gsm] = listing(
                    f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm[:-3]}nnn/{gsm}/suppl/"
                )
            except Exception as e:  # noqa: BLE001
                sample_sizes[gsm] = str(e)
        entry["example_sample_files"] = sample_sizes
        out[acc] = entry
    os.makedirs(os.path.join(HERE, "results_track_b"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results_track_b", "geo_sizes.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
