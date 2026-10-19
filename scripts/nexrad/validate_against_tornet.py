"""
Reproduces released TorNet samples from raw NEXRAD archives and reports how closely
the conversion pipeline matches them, variable by variable.

For each TorNet .nc file and each of its frames, this script
  1. downloads the Level-II volume nearest the frame time (AWS),
  2. downloads the matching Level-III N0K/NAK KDP products
     (AWS for 2020+, otherwise the Google Cloud daily Level-III archive),
  3. runs ``level2_to_tornet_volume`` + ``extract_chip`` with the file's chip limits,
  4. compares every gate with the TorNet file.

Example:
  python scripts/nexrad/validate_against_tornet.py $TORNET_ROOT/train/2013/*.nc \
      --cache-dir /tmp/nexrad_cache --max-files 20 --kdp-source both
"""
import argparse
import datetime
import glob
import io
import json
import os
import re
import sys
import tarfile
import urllib.request

import numpy as np
import xarray as xr

from tornet.data.constants import ALL_VARIABLES
from tornet.nexrad import fetch
from tornet.nexrad import level2 as l2
from tornet.nexrad.dealias import UNetVDA
from tornet.nexrad.pipeline import extract_chip, level2_to_tornet_volume

GCS_L3 = "https://storage.googleapis.com/gcp-public-data-nexrad-l3"


def cached(cache_dir, name, getter):
    path = os.path.join(cache_dir, name)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = getter()
        with open(path + ".tmp", "wb") as f:
            f.write(data)
        os.replace(path + ".tmp", path)
    with open(path, "rb") as f:
        return f.read()


def gcs_level3_kdp(cache_dir, site, day, products=("N0K", "NAK")):
    """Extracts KDP products of one site-day from the GCS Level-III archive (cached)."""
    out_dir = os.path.join(cache_dir, "l3", site, f"{day:%Y%m%d}")
    marker = os.path.join(out_dir, ".done")
    if not os.path.exists(marker):
        os.makedirs(out_dir, exist_ok=True)
        base = f"{GCS_L3}/{day:%Y/%m/%d}/{site}/NWS_NEXRAD_NXL3_{site}_{day:%Y%m%d}000000_{day:%Y%m%d}235959"
        for ext in (".tar.gz", ".tar.Z", ".tar"):
            try:
                resp = urllib.request.urlopen(base + ext, timeout=600)
                break
            except Exception:
                resp = None
        if resp is None:
            raise FileNotFoundError(f"No GCS Level-III archive for {site} {day}")
        mode = "r|gz" if ext == ".tar.gz" else "r|"
        with tarfile.open(fileobj=resp, mode=mode) as tf:
            for m in tf:
                if any(f"_{p}{site[1:]}_" in m.name for p in products) and m.isfile():
                    with open(os.path.join(out_dir, os.path.basename(m.name)), "wb") as f:
                        f.write(tf.extractfile(m).read())
        open(marker, "w").close()
    return out_dir


def find_gcs_kdp(cache_dir, site, product, t):
    files = []
    for day in {(t - datetime.timedelta(minutes=5)).date(), t.date()}:
        d = gcs_level3_kdp(cache_dir, site, day)
        files += glob.glob(os.path.join(d, f"*_{product}{site[1:]}_*"))
    if not files:
        return None
    def ftime(fn):
        return datetime.datetime.strptime(re.search(r"_(\d{12})$", fn).group(1), "%Y%m%d%H%M")
    # GCS names are truncated to the minute
    best = min(files, key=lambda fn: abs(ftime(fn) - t.replace(second=0, microsecond=0)))
    if abs(ftime(best) - t.replace(second=0, microsecond=0)) > datetime.timedelta(seconds=60):
        return None
    with open(best, "rb") as f:
        return f.read()


def compare(a, b):
    """Fraction of gates equal (NaN==NaN) and within tolerance."""
    nan_eq = ~np.isfinite(a) & ~np.isfinite(b)
    both = np.isfinite(a) & np.isfinite(b)
    exact = (nan_eq | (both & (a == b))).mean()
    close = (nan_eq | (both & (np.abs(a - b) <= 1e-3 + 1e-3 * np.abs(b)))).mean()
    return float(exact), float(close)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/tornet/nexrad"))
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--last-frame-only", action="store_true")
    ap.add_argument("--kdp-source", choices=["level3", "level2", "both"], default="level3")
    ap.add_argument("--selection", choices=["nearest", "latest", "first"], default="nearest",
                    help="cut selection; TorNet used the cut nearest each frame time")
    ap.add_argument("--dealias-weights", default=None, help=".npz weights for U-Net VDA")
    ap.add_argument("--no-dealias", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    files = sorted(args.files)[: args.max_files]
    dealiaser = None if args.no_dealias else UNetVDA.from_pretrained(args.dealias_weights)
    results = []
    for fn in files:
        ds = xr.open_dataset(fn)
        site = ds.attrs["site_name"]
        frames = range(ds.sizes["time"])
        if args.last_frame_only:
            frames = [ds.sizes["time"] - 1]
        for fr in frames:
            t = datetime.datetime.utcfromtimestamp(ds.time.values[fr].astype("datetime64[s]").astype(int))
            rec = {"file": os.path.basename(fn), "frame": fr, "site": site}
            try:
                key = fetch.find_level2_volume(site, t)
                # TorNet chose each cut nearest in time to the frame; offer the
                # neighbouring volumes as candidates too.
                keys = fetch.list_level2_volumes(site, fetch.level2_key_time(key).date())
                i = keys.index(key) if key in keys else None
                cand = [key] if i is None else keys[max(i - 1, 0): i + 2]
                vols = [l2.open_level2(cached(args.cache_dir, "l2/" + k,
                                              lambda k=k: fetch.download(fetch.LEVEL2_BUCKET, k))) for k in cand]
                rec["level2"] = key
                rec["vcp"] = vols[0].vcp
                l3_found = {}

                def l3_fetch(site_, product, cut_time):
                    if cut_time.year >= 2020:
                        k = fetch.find_level3_product(site_, product, cut_time)
                        data = cached(args.cache_dir, "l3aws/" + k,
                                      lambda: fetch.download(fetch.LEVEL3_BUCKET, k)) if k else None
                    else:
                        data = find_gcs_kdp(args.cache_dir, site_, product, cut_time)
                    l3_found[product] = data is not None
                    return data

                lims = dict(az_lower=float(ds.azimuth_limits[0]), rng_lower=float(ds.range_limits[0]))
                sources = ["level3", "level2"] if args.kdp_source == "both" else [args.kdp_source]
                for src in sources:
                    tv = level2_to_tornet_volume(vols, level3_kdp=l3_fetch if src == "level3" else None,
                                                 kdp_source="auto" if src == "level3" else "level2",
                                                 dealiaser=dealiaser, selection=args.selection,
                                                 target_time=t)
                    rec["level3_found"] = l3_found
                    chip = extract_chip(tv, **lims)
                    vars_ = ALL_VARIABLES if src == "level3" else ["KDP"]
                    for v in vars_:
                        tag = v if src == "level3" else "KDP_from_level2"
                        if v == "KDP" and src == "level3" and not all(l3_found.values()):
                            continue
                        ex, cl = compare(chip[v][0], ds[v].values[fr])
                        rec[tag] = {"exact": ex, "close": cl}
                        if v == "KDP" and src == "level2":
                            ref, est = ds[v].values[fr], chip[v][0]
                            m = np.isfinite(ref) & np.isfinite(est)
                            vr, ve = np.isfinite(ref), np.isfinite(est)
                            rec[tag].update({
                                "mae": float(np.abs(ref[m] - est[m]).mean()) if m.any() else None,
                                "corr": float(np.corrcoef(ref[m], est[m])[0, 1]) if m.sum() > 2 else None,
                                "valid_iou": float((vr & ve).sum() / max((vr | ve).sum(), 1)),
                            })
                    if src == "level3":
                        rf = chip["range_folded_mask"][0] == ds["range_folded_mask"].values[fr]
                        rec["range_folded_mask"] = {"exact": float(rf.mean()), "close": float(rf.mean())}
                        rec["nyquist_match"] = bool(np.allclose(tv.nyquist, ds.nyquist_velocity.values[fr], atol=0.01))
                        rec["elevation"] = tv.elevations.tolist()
            except Exception as e:  # keep going; report the failure
                rec["error"] = f"{type(e).__name__}: {e}"
            results.append(rec)
            summary = " ".join(f"{k}={v['exact']:.4f}" for k, v in rec.items() if isinstance(v, dict) and "exact" in v)
            print(f"{rec['file']} f{fr} {summary} {rec.get('error', '')}", flush=True)

    # Aggregate
    print("\n=== Mean fraction of gates matching exactly (and within 1e-3) ===")
    keys = sorted({k for r in results for k, v in r.items() if isinstance(v, dict) and "exact" in v})
    for k in keys:
        ex = [r[k]["exact"] for r in results if k in r]
        cl = [r[k]["close"] for r in results if k in r]
        print(f"{k:22s} n={len(ex):4d} exact={np.mean(ex):.5f} close={np.mean(cl):.5f} min_exact={np.min(ex):.5f}")
    if any("KDP_from_level2" in r for r in results):
        for stat in ("mae", "corr", "valid_iou"):
            vals = [r["KDP_from_level2"][stat] for r in results if "KDP_from_level2" in r and r["KDP_from_level2"][stat] is not None]
            print(f"KDP_from_level2 {stat}: {np.mean(vals):.4f}")
    errs = [r for r in results if "error" in r]
    print(f"errors: {len(errs)} / {len(results)}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=1, default=str)


if __name__ == "__main__":
    sys.exit(main())
