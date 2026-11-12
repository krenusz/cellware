"""Crawl the Raj lab 'Paper' Dropbox share: metadata only (path, Dropbox file id, size, modified).
Never downloads file contents. Output: one JSON line per file/folder."""
import json, re, subprocess, sys, time, urllib.parse
sys.stdout.reconfigure(encoding="utf-8")
OUT, JAR = sys.argv[1], sys.argv[2]
LK, RL, ROOT = "yklfs2p4s9dg8uyb6w8nt", "gl8drm3l79j9k7oi1wuro4s6w", "ADXsVD7ABiklQnTyCSmhU5Q"
UA = "Mozilla/5.0"
subprocess.run(["curl", "-s", "-L", "-c", JAR, "-A", UA, "-o", "/dev/null",
                f"https://www.dropbox.com/scl/fo/{LK}/{ROOT}?rlkey={RL}&dl=0"])
T = [l.split("\t")[6].strip() for l in open(JAR) if "\tt\t" in l][0]

def post(secure_hash, sub_path, voucher=None):
    d = {"is_xhr": "true", "t": T, "link_key": LK, "link_type": "c", "secure_hash": secure_hash,
         "sub_path": sub_path, "rlkey": RL}
    if voucher: d["voucher"] = voucher
    args = ["curl", "-s", "-b", JAR, "-A", UA, "-X", "POST", "https://www.dropbox.com/list_shared_link_folder_entries"]
    for k, v in d.items(): args += ["--data-urlencode", f"{k}={v}"]
    for wait in (0, 5, 15, 45, 120, 300):
        time.sleep(wait)
        r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
        try: return json.loads(r)
        except Exception: pass
    raise RuntimeError(f"failed {sub_path}: {r[:200]}")

out = open(OUT, "w", encoding="utf-8")
stack, nf, nd = [(ROOT, "")], 0, 0
while stack:
    h, sp = stack.pop()
    voucher = None
    while True:
        j = post(h, sp, voucher)
        for e in j.get("entries", []):
            path = (sp + "/" + e["filename"]).lstrip("/")
            m = re.search(rf"/scl/fo/{LK}/([^/]+)/(.*?)\?", e.get("href", ""))
            rec = {"path": path, "is_dir": e.get("is_dir", False), "bytes": e.get("bytes"), "ts": e.get("ts"),
                   "id": e.get("file_id") or e.get("sjid") or e.get("folder_id"), "href": e.get("href")}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rec["is_dir"]:
                nd += 1
                stack.append((m.group(1), "/" + urllib.parse.unquote(m.group(2))) if m else (h, "/" + path))
            else:
                nf += 1
        out.flush()
        voucher = j.get("next_request_voucher")
        if not (j.get("has_more_entries") and voucher): break
    if (nd + nf) % 200 < 50: print("dirs", nd, "files", nf, flush=True)
print("DONE dirs", nd, "files", nf, flush=True)
