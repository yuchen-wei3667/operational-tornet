"""
Benchmark full-sweep Level-II -> model-ready TorNet inputs (and optionally inference).

Times each stage of ``level2_to_model_inputs`` per volume: Level-II decode, cut
selection + moment decoding, U-Net VDA dealiasing, KDP (Level-II estimate or
Level-III decode), netCDF quantization, chip tiling + coordinates, and model inference.
Files are read into memory first so disk I/O is excluded; Level-III products are
fetched once before timing (fetch latency is reported separately).  With
``--download``, each Level-II volume is also re-downloaded from the AWS archive
(key derived from its file name) and ``full_scan_s`` = download + Level-III fetch +
conversion + inference, i.e. the wall-clock time from a scan on AWS to probabilities.

Example:
  python scripts/nexrad/benchmark_level2.py data/*V06 --repeat 3 --kdp-source both --infer
  python scripts/nexrad/benchmark_level2.py data/*V06 --kdp-source level3 --infer --download
"""
import argparse
import collections
import contextlib
import json
import os
import resource
import statistics
import time

import numpy as np

from tornet.nexrad import fetch
from tornet.nexrad import level2 as l2
from tornet.nexrad import pipeline
from tornet.nexrad.model_inputs import default_dealiaser, iter_model_input_batches, volume_to_model_inputs

TIMES = collections.defaultdict(float)


def _timed(name, fn):
    def wrapper(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            TIMES[name] += time.perf_counter() - t
    return wrapper


@contextlib.contextmanager
def instrument():
    """Temporarily wraps the pipeline's stage functions with timers."""
    patches = [(l2, "open_level2", "decode_level2"), (l2, "build_tilt", "build_tilts"),
               (pipeline, "kdp_from_level2", "kdp_level2"), (pipeline, "decode_level3_kdp", "kdp_level3"),
               (pipeline, "level3_to_tornet_grid", "kdp_regrid"), (pipeline, "netcdf_quantize", "quantize")]
    saved = [(m, a, getattr(m, a)) for m, a, _ in patches]
    for m, a, name in patches:
        setattr(m, a, _timed(name, getattr(m, a)))
    try:
        yield
    finally:
        for m, a, fn in saved:
            setattr(m, a, fn)


def level2_archive_key(path):
    """SITEYYYYMMDD_HHMMSS_V06[.gz] -> YYYY/MM/DD/SITE/<name> in the Level-II archive."""
    name = os.path.basename(path)
    return f"{name[4:8]}/{name[8:10]}/{name[10:12]}/{name[:4]}/{name}"


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("level2_files", nargs="+")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--kdp-source", choices=["level2", "level3", "both"], default="level2")
    ap.add_argument("--max-range", type=float, default=230.0, help="km")
    ap.add_argument("--infer", action="store_true", help="also time the TorNet model on all chips")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--download", action="store_true",
                    help="also time downloading each volume from the AWS Level-II archive")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    t = time.perf_counter()
    vda = default_dealiaser()
    startup = {"load_dealiaser": time.perf_counter() - t}
    model = None
    if args.infer:
        from tornet.nexrad.inference import load_tornet_model
        t = time.perf_counter()
        model = load_tornet_model()
        startup["load_model"] = time.perf_counter() - t
    dealias = _timed("dealias", vda)

    sources = ["level2", "level3"] if args.kdp_source == "both" else [args.kdp_source]
    rows = []
    for path in args.level2_files:
        with open(path, "rb") as f:
            raw = f.read()
        head = l2.open_level2(raw)
        download_s = None
        if args.download:
            t = time.perf_counter()
            fetch.download(fetch.LEVEL2_BUCKET, level2_archive_key(path))
            download_s = time.perf_counter() - t
        l3 = None
        fetch_s = None
        if "level3" in sources:
            t = time.perf_counter()
            l3 = []
            for i, elev in enumerate(l2.TORNET_TILTS):
                cut_time = l2.build_tilt(head, elev, 1).time
                l3.append(fetch.level3_kdp_fetcher(head.site, pipeline.KDP_PRODUCT_FOR_TILT[i], cut_time))
            fetch_s = time.perf_counter() - t
            if any(x is None for x in l3):
                print(f"{os.path.basename(path)}: Level-III KDP not found, skipping level3 mode")
        for src in sources:
            if src == "level3" and (l3 is None or any(x is None for x in l3)):
                continue
            for rep in range(args.repeat):
                TIMES.clear()
                t_all = time.perf_counter()
                with instrument():
                    vol = pipeline.level2_to_tornet_volume(
                        raw, level3_kdp=l3 if src == "level3" else None, kdp_source=src, dealiaser=dealias)
                t = time.perf_counter()
                mi = volume_to_model_inputs(vol, rng_max=args.max_range * 1000.0)
                TIMES["model_inputs"] = time.perf_counter() - t
                total = time.perf_counter() - t_all
                n_chips = len(mi)
                del mi
                if model is not None:
                    # production path: stream batches of chips straight into the model
                    keys = list(model.input.keys())
                    t_tile = t_pred = 0.0
                    t = time.perf_counter()
                    for b in iter_model_input_batches(vol, args.batch_size, rng_max=args.max_range * 1000.0):
                        t_tile += time.perf_counter() - t
                        t = time.perf_counter()
                        model.predict_on_batch({k: b.inputs[k] for k in keys})
                        t_pred += time.perf_counter() - t
                        t = time.perf_counter()
                    TIMES["streamed_model_inputs"] = t_tile
                    TIMES["inference"] = t_pred
                st = dict(TIMES)
                st["other"] = total - sum(v for k, v in st.items() if k not in ("inference", "streamed_model_inputs"))
                row = {"file": os.path.basename(path), "site": vol.site, "vcp": vol.vcp,
                       "kdp": src, "rep": rep, "n_gates": vol.n_gates, "n_chips": n_chips,
                       "size_mb": len(raw) / 1e6, "convert_s": total,
                       "convert_streamed_s": (total - st["model_inputs"] + st["streamed_model_inputs"]
                                              if model is not None else None),
                       "end_to_end_s": (total - st["model_inputs"] + st["streamed_model_inputs"] + st["inference"]
                                        if model is not None else total),
                       "level3_fetch_s": fetch_s if src == "level3" else None,
                       "download_s": download_s,
                       "stages": st, "peak_rss_mb": peak_rss_mb()}
                if download_s is not None and model is not None:
                    row["full_scan_s"] = row["end_to_end_s"] + download_s + (row["level3_fetch_s"] or 0.0)
                rows.append(row)
                print(f"{row['file']:28s} vcp={vol.vcp:<3d} kdp={src:6s} rep={rep} "
                      f"convert={total:6.2f}s" + (f" end_to_end={row['end_to_end_s']:6.2f}s" if model else "") +
                      "  " + " ".join(f"{k}={v:.2f}" for k, v in st.items()))

    # warm runs only (rep > 0) when repeating, so one-off graph tracing is excluded
    warm = [r for r in rows if r["rep"] > 0] or rows
    print("\nSummary (median over warm runs; startup excluded)")
    print(f"  startup: " + ", ".join(f"{k}={v:.1f}s" for k, v in startup.items()))
    summary = {"startup_s": startup, "cpu_count": os.cpu_count(), "by_kdp_source": {}}
    for src in sources:
        rs = [r for r in warm if r["kdp"] == src]
        if not rs:
            continue
        stage_names = sorted({k for r in rs for k in r["stages"]})
        med = {k: statistics.median(r["stages"].get(k, 0.0) for r in rs) for k in stage_names}
        s = {"n_runs": len(rs), "n_volumes": len({r["file"] for r in rs}),
             "convert_s_median": statistics.median(r["convert_s"] for r in rs),
             "convert_s_p90": float(np.percentile([r["convert_s"] for r in rs], 90)),
             "convert_s_max": max(r["convert_s"] for r in rs),
             "end_to_end_s_median": statistics.median(r["end_to_end_s"] for r in rs),
             "n_chips_median": statistics.median(r["n_chips"] for r in rs),
             "stages_median_s": med}
        streamed = [r["convert_streamed_s"] for r in rs if r["convert_streamed_s"] is not None]
        if streamed:
            s["convert_streamed_s_median"] = statistics.median(streamed)
            s["convert_streamed_s_p90"] = float(np.percentile(streamed, 90))
        for k in ("download_s", "full_scan_s"):
            vals = [r[k] for r in rs if r.get(k) is not None]
            if vals:
                s[f"{k}_median"] = statistics.median(vals)
                s[f"{k}_p90"] = float(np.percentile(vals, 90))
        fetches = [r["level3_fetch_s"] for r in rs if r["level3_fetch_s"] is not None]
        if fetches:
            s["level3_fetch_s_median"] = statistics.median(fetches)
        summary["by_kdp_source"][src] = s
        print(f"  kdp={src}: {s['n_volumes']} volumes, convert median {s['convert_s_median']:.2f}s "
              f"(p90 {s['convert_s_p90']:.2f}s, max {s['convert_s_max']:.2f}s), "
              f"end-to-end median {s['end_to_end_s_median']:.2f}s")
        if streamed:
            print(f"    streamed (production path, batches into the model): convert median "
                  f"{s['convert_streamed_s_median']:.2f}s (p90 {s['convert_streamed_s_p90']:.2f}s)")
        if "full_scan_s_median" in s:
            print(f"    full scan (download + Level-III fetch + convert + inference): median "
                  f"{s['full_scan_s_median']:.2f}s (p90 {s['full_scan_s_p90']:.2f}s); "
                  f"Level-II download median {s['download_s_median']:.2f}s")
        print("    stages: " + ", ".join(f"{k}={v:.2f}s" for k, v in sorted(med.items(), key=lambda kv: -kv[1])))
    summary["peak_rss_mb"] = peak_rss_mb()
    print(f"  peak RSS {summary['peak_rss_mb']:.0f} MB, {os.cpu_count()} CPUs")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"summary": summary, "runs": rows}, f, indent=1)


if __name__ == "__main__":
    main()
