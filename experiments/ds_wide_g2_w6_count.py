"""DS-WIDE-G2 stage 3 structure count, rules of amendments 3 (7eafa32) and 4 (1444cb9). Columns D (Time) and
E (TrackID) only, row 3 down. openpyxl read_only parses whole rows internally; only D and E are returned to this
code.

 R0 a row counts if Time or TrackID is non-blank; blank or non-numeric values are left out of track logic;
    integral floats are integers.
 Hold: rows stay >= the new level at 5 adjacent sheet frames (sheet's sorted distinct Times), the rise frame
    first; a missing frame in the track breaks it. A rise is read between the track's consecutive observed Times,
    also across a gap.
 R1 pairs_R1: each held one-step rise (k to k+1), including a held re-rise after a fall. A track whose first Time
    already has >= 2 rows is counted under starts_ge2; that start is not a pair, a later held one-step rise is.
    trees_with_pair_R1: tracks with >= 1 such pair.
    pairs_R1_strict: a held one-step rise counts only if its new level is above max_held, the highest level the
    track has already held. max_held begins at the starting level if that start meets the hold (rows >= it at 5
    adjacent sheet frames from the first Time), else at 0; every later held rise (jumps included) raises it.
 Held jumps: a held step of >= 2 rows adds to held_jumps (and held_jumps_after_pair if the track already has an
    R1 pair). Never an R1 pair.
 R2: a one-step rise ending at >= 3 rows in a track with an R1 pair at an earlier Time. with_hold needs this rise
    held; without_hold drops the hold on this rise only; with_hold_strict needs this rise to be strict (held, new
    level above max_held) and a strict R1 pair at an earlier Time in the track.
 R3 quality flag: rows per Time go up and later go down along the track's own Times; a gap or the track ending
    is not a fall.
 R1-R3 and held jumps for the seven-sheet (control) group only; the other sheets keep sets of TrackIDs and Times
    only.

Usage: w6_count.py --selftest            (made-up tracks only; opens no file)
       w6_count.py WORKBOOK OUT_JSON     (OUT_JSON must not exist)
Output is held to the end; the watchdog prints one stop line."""
import ctypes
import json
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from ctypes import wintypes

LIMIT_S = 180
HOLD = 5
COLS = ("pairs_R1", "pairs_R1_strict", "trees_with_pair_R1", "starts_ge2", "rises_R2_without_hold",
        "rises_R2_with_hold", "rises_R2_with_hold_strict", "held_jumps", "held_jumps_after_pair",
        "tracks_fall_after_rise_R3")


def key(v):
    """-> (type label, numeric key or None)"""
    if v is None or (isinstance(v, str) and not v.strip()):
        return "blank", None
    if isinstance(v, bool):
        return "bool", None
    if isinstance(v, int):
        return "int", v
    if isinstance(v, float):
        if v.is_integer():
            return "integral_float", int(v)
        return "other_float", v
    return type(v).__name__, None


def track_stats(per, frames):
    index = {f: i for i, f in enumerate(frames)}
    n = dict.fromkeys(COLS, 0)
    for by_time in per.values():
        times = sorted(by_time)

        def held(tm, level):
            i = index[tm]
            window = frames[i:i + HOLD]
            return len(window) == HOLD and all(by_time.get(f, 0) >= level for f in window)

        n["starts_ge2"] += by_time[times[0]] >= 2
        pairs = strict_pairs = 0
        start = by_time[times[0]]
        max_held = start if held(times[0], start) else 0
        rose = fell = False
        for a, b in zip(times, times[1:]):
            before, after = by_time[a], by_time[b]
            if after > before:
                rose = True
                is_held = held(b, after)
                one_step = after == before + 1
                new_level = after > max_held
                if one_step and after >= 3 and pairs:  # pairs holds earlier Times only at this point
                    n["rises_R2_without_hold"] += 1
                    n["rises_R2_with_hold"] += is_held
                    n["rises_R2_with_hold_strict"] += is_held and new_level and strict_pairs > 0
                if is_held:
                    if one_step:
                        pairs += 1
                        strict_pairs += new_level
                        n["pairs_R1_strict"] += new_level
                    else:
                        n["held_jumps"] += 1
                        n["held_jumps_after_pair"] += pairs > 0
                    max_held = max(max_held, after)
            elif after < before and rose:
                fell = True
        n["pairs_R1"] += pairs
        n["trees_with_pair_R1"] += pairs > 0
        n["tracks_fall_after_rise_R3"] += fell
    return n


def selftest():
    frames = list(range(1, 21))

    def one(spec):
        return tuple(track_stats({0: spec}, frames)[c] for c in COLS)

    def run(*parts):
        return {t: k for lo, hi, k in parts for t in range(lo, hi + 1)}

    # expected, in COLS order:
    #   R1, R1 strict, trees, starts_ge2, R2 no hold, R2 hold, R2 strict, held jumps, jumps after pair, R3
    cases = {
        "held rise": (run((1, 1, 1), (2, 6, 2)), (1, 1, 1, 0, 0, 0, 0, 0, 0, 0)),
        "cut by track end": (run((1, 1, 1), (2, 5, 2)), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
        "cut by gap": (run((1, 1, 1), (2, 4, 2), (6, 8, 2)), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
        "start at 2": (run((1, 6, 2)), (0, 0, 0, 1, 0, 0, 0, 0, 0, 0)),
        "start at 2 then held rise to 3": (run((1, 3, 2), (4, 8, 3)), (1, 1, 1, 1, 0, 0, 0, 0, 0, 0)),
        "start at 2, dropout, back to 2": (run((1, 5, 2), (6, 6, 1), (7, 11, 2)), (1, 0, 1, 1, 0, 0, 0, 0, 0, 0)),
        "start at 2, dropout, back to 2, held rise to 3": (run((1, 5, 2), (6, 6, 1), (7, 11, 2), (12, 16, 3)),
                                                           (2, 1, 1, 1, 1, 1, 0, 0, 0, 0)),
        "mid-track dropout, then held rise to 3": (run((1, 1, 1), (2, 6, 2), (7, 7, 1), (8, 12, 2), (13, 17, 3)),
                                                   (3, 2, 1, 0, 1, 1, 1, 0, 0, 1)),
        "start at 2 shorter than the hold": (run((1, 3, 2), (4, 4, 1), (5, 9, 2)),
                                             (1, 1, 1, 1, 0, 0, 0, 0, 0, 0)),
        "two held rises": (run((1, 1, 1), (2, 6, 2), (7, 11, 3)), (2, 2, 1, 0, 1, 1, 1, 0, 0, 0)),
        "second rise inside first hold": (run((1, 1, 1), (2, 2, 2), (3, 8, 3)), (2, 2, 1, 0, 1, 1, 1, 0, 0, 0)),
        "R2 without hold only": (run((1, 1, 1), (2, 6, 2), (7, 7, 3), (8, 8, 2)), (1, 1, 1, 0, 1, 0, 0, 0, 0, 1)),
        "held jump, no pair": (run((1, 1, 1), (2, 8, 3)), (0, 0, 0, 0, 0, 0, 0, 1, 0, 0)),
        "unheld jump": (run((1, 1, 1), (2, 4, 3)), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
        "jump to 4 after a pair": (run((1, 1, 1), (2, 6, 2), (7, 11, 4)), (1, 1, 1, 0, 0, 0, 0, 1, 1, 0)),
        "rise across gap": (run((1, 1, 1), (3, 7, 2)), (1, 1, 1, 0, 0, 0, 0, 0, 0, 0)),
        "fall after unheld rise": (run((1, 1, 1), (2, 2, 2), (3, 4, 1)), (0, 0, 0, 0, 0, 0, 0, 0, 0, 1)),
        "fall with no rise": (run((1, 1, 2), (2, 3, 1)), (0, 0, 0, 1, 0, 0, 0, 0, 0, 0)),
        "unheld first rise, then jump to 3": (run((1, 1, 1), (2, 2, 2), (3, 3, 1), (4, 9, 3)),
                                              (0, 0, 0, 0, 0, 0, 0, 1, 0, 1)),
        "dropout 2-1-2": (run((1, 1, 1), (2, 6, 2), (7, 7, 1), (8, 12, 2)), (2, 1, 1, 0, 0, 0, 0, 0, 0, 1)),
        "R2 strict on a re-rise to 3": (run((1, 1, 1), (2, 6, 2), (7, 11, 3), (12, 12, 2), (13, 17, 3)),
                                        (3, 2, 1, 0, 2, 2, 1, 0, 0, 1)),
        "held jump, then lower rise is not strict": (run((1, 1, 1), (2, 6, 3), (7, 7, 1), (8, 12, 2)),
                                                     (1, 0, 1, 0, 0, 0, 0, 1, 0, 1)),
    }
    bad = {name: (one(spec), want) for name, (spec, want) in cases.items() if one(spec) != want}
    assert not bad, bad
    both = track_stats({1: cases["dropout 2-1-2"][0], 2: cases["jump to 4 after a pair"][0]}, frames)
    assert tuple(both[c] for c in COLS) == (3, 2, 2, 0, 0, 0, 0, 1, 1, 1), both
    assert key(2.0) == ("integral_float", 2) and key(" ") == ("blank", None) and key("a") == ("str", None)
    return len(cases)


def peak_ram_mb():
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                           "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                           "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(pmc)
    ok = k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return round(pmc.PeakWorkingSetSize / 2**20) if ok else None


def count(path, out):
    t0 = time.perf_counter()

    def expired():
        print(f"WATCHDOG: {LIMIT_S} s passed, stopped with no output", flush=True)
        os._exit(99)

    timer = threading.Timer(LIMIT_S, expired)
    timer.daemon = True
    timer.start()
    if os.path.exists(out):
        sys.exit("output already exists: stop (one run only)")

    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    names = wb.sheetnames
    words, labels = {}, []
    for name in names:
        labels.append(re.sub(r"[A-Za-z0-9]*[A-Za-z][A-Za-z0-9]*",
                             lambda m: words.setdefault(m.group(0), f"w{len(words) + 1}"), name))
    group_size = Counter(label.split("_")[0] for label in labels)
    seven = [g for g, k in group_size.items() if k == 7]
    assert len(seven) == 1
    seven = seven[0]

    sheets = []
    for pos, (name, label) in enumerate(zip(names, labels), 1):
        ws = wb[name]
        ws.reset_dimensions()
        head = next(ws.iter_rows(min_row=2, max_row=2, min_col=4, max_col=5, values_only=True))
        assert tuple(str(h).strip() for h in head) == ("Time", "TrackID"), "unexpected header"
        in_group = label.split("_")[0] == seven
        n_rows = 0
        tracks, frames = set(), set()
        types_time, types_track = Counter(), Counter()
        per = defaultdict(Counter)
        for tv, kv in ws.iter_rows(min_row=3, min_col=4, max_col=5, values_only=True):
            tt, tm = key(tv)
            kt, tr = key(kv)
            if tt == "blank" and kt == "blank":
                continue
            n_rows += 1
            types_time[tt] += 1
            types_track[kt] += 1
            if tm is not None:
                frames.add(tm)
            if tr is not None:
                tracks.add(tr)
            if in_group and tm is not None and tr is not None:
                per[tr][tm] += 1
        e = {"position": pos, "sheet": label, "rows": n_rows, "distinct_tracks": len(tracks),
             "time_min": min(frames) if frames else None, "time_max": max(frames) if frames else None,
             "distinct_times": len(frames)}
        if set(types_time) - {"int"}:
            e["time_type_counts"] = dict(types_time)
        if set(types_track) - {"int"}:
            e["track_type_counts"] = dict(types_track)
        if in_group:
            e.update(track_stats(per, sorted(frames)))
        sheets.append(e)
        del per
    wb.close()
    try:
        ram = peak_ram_mb()
    except Exception:  # a RAM readout must not cost the one run
        ram = None
    res = {"control_group": seven, "sheets": sheets,
           "control_group_totals": {c: sum(s.get(c, 0) for s in sheets) for c in COLS},
           "wall_s": round(time.perf_counter() - t0, 1), "peak_ram_mb": ram}
    with open(out, "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    timer.cancel()
    for s in sheets:
        print(json.dumps(s))
    print("totals", json.dumps(res["control_group_totals"]), "| wall_s", res["wall_s"], "| peak_ram_mb", ram)


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        n_cases = selftest()
        ram = peak_ram_mb()
        assert ram and ram > 0, ram
        import openpyxl

        print(f"selftest passed: {n_cases} made-up track cases, plus two-track totals and type keys; "
              f"peak RAM readout works ({ram} MB); openpyxl {openpyxl.__version__}")
        sys.exit(0)
    if len(sys.argv) != 3:
        sys.exit("usage: --selftest | WORKBOOK OUT_JSON")
    selftest()
    count(sys.argv[1], sys.argv[2])
