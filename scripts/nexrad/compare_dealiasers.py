"""
Compares velocity dealiasers on released TorNet samples: how closely each reproduces
TorNet's stored VEL (dealiased with U-Net VDA) and how much it moves the TorNet
model's probability.

For each TorNet frame the Level-II volume is converted with each dealiaser; the chip's
VEL is compared with the file's VEL, and the model is run on the file's own sample
with only VEL swapped, so differences come from dealiasing alone.  Frames whose
U-Net VEL matches the file on < 98% of gates are left out of the summary (the
cut/volume was not reproduced, unrelated to dealiasing).

Dealiasers: unet_vda (TorNet's), none, pyart_region (Py-ART region-based).
Py-ART's 2-D phase unwrapping is not included: it took > 10 min per super-res sweep.

Example:
  python scripts/nexrad/compare_dealiasers.py $TORNET_ROOT/train/2022/*.nc \
      --cache-dir /tmp/nexrad_cache --max-files 27 --json-out dealias.json
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import xarray as xr

from tornet.nexrad import fetch
from tornet.nexrad import level2 as l2
from tornet.nexrad.inference import load_tornet_model
from tornet.nexrad.model_inputs import chip_to_model_input, default_dealiaser
from tornet.nexrad.pipeline import extract_chip, level2_to_tornet_volume

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_against_tornet import cached  # noqa: E402

CHIP_VARS = ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH", "range_folded_mask"]


def pyart_region(vel, nyquist):
    """Py-ART region-based dealiasing of one [n_az, n_rng] sweep."""
    import pyart
    n_az, n = vel.shape
    radar = pyart.testing.make_empty_ppi_radar(n, n_az, 1)
    radar.azimuth["data"] = (np.arange(n_az) + 0.5) * 360.0 / n_az
    radar.range["data"] = 2125.0 + 250.0 * np.arange(n)
    radar.instrument_parameters = {"nyquist_velocity": {"data": np.full(n_az, nyquist, np.float32)}}
    radar.add_field("velocity", {"data": np.ma.masked_invalid(vel)})
    out = pyart.correct.dealias_region_based(radar, nyquist_vel=nyquist, rays_wrap_around=True)
    return np.ma.filled(out["data"].astype(np.float32), np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/tornet/nexrad"))
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    dealiasers = {"unet_vda": default_dealiaser(), "none": None, "pyart_region": pyart_region}
    times = {k: [] for k in dealiasers}

    def timed(name, fn):
        def wrapper(vel, nyquist):
            t = time.perf_counter()
            out = fn(vel, nyquist)
            times[name].append(time.perf_counter() - t)
            return out
        return wrapper

    model = load_tornet_model()
    keys = list(model.input.keys())
    rows = []
    for fn in sorted(args.files)[: args.max_files]:
        ds = xr.open_dataset(fn)
        site = ds.attrs["site_name"]
        for fr in range(ds.sizes["time"]):
            t = datetime.datetime.utcfromtimestamp(ds.time.values[fr].astype("datetime64[s]").astype(int))
            try:
                key = fetch.find_level2_volume(site, t)
                ks = fetch.list_level2_volumes(site, fetch.level2_key_time(key).date())
                i = ks.index(key) if key in ks else None
                cand = [key] if i is None else ks[max(i - 1, 0): i + 2]
                vols = [l2.open_level2(cached(args.cache_dir, "l2/" + k,
                                              lambda k=k: fetch.download(fetch.LEVEL2_BUCKET, k))) for k in cand]
            except Exception as e:  # keep going; report the failure
                print(f"{os.path.basename(fn)} f{fr} skipped: {type(e).__name__}: {e}")
                continue
            lims = dict(az_lower=float(ds.azimuth_limits[0]), rng_lower=float(ds.range_limits[0]))
            ref = ds["VEL"].values[fr]
            sample = None
            rec = {"file": os.path.basename(fn), "frame": fr}
            for name, d in dealiasers.items():
                vol = level2_to_tornet_volume(vols, dealiaser=None if d is None else timed(name, d),
                                              kdp_source="level2", selection="nearest", target_time=t)
                chip = extract_chip(vol, **lims)
                if sample is None:
                    # the TorNet sample itself; only VEL is swapped below
                    sample = dict(chip)
                    sample.update({v: ds[v].values[fr][None].astype(np.float32) for v in CHIP_VARS})
                vel = chip["VEL"][0]
                both = np.isfinite(ref) & np.isfinite(vel)
                diff = np.abs(vel[both] - ref[both])
                s = dict(sample, VEL=vel[None])
                x = chip_to_model_input(s, keys=keys)
                logit = np.asarray(model.predict_on_batch({k: x[k][None] for k in keys})).ravel()[0]
                rec[name] = {"match": float((diff <= 0.11).mean()) if both.any() else 1.0,
                             "fold_err": float((diff > 5.0).mean()) if both.any() else 0.0,
                             "prob": float(1.0 / (1.0 + np.exp(-logit)))}
            x = chip_to_model_input(sample, keys=keys)
            logit = np.asarray(model.predict_on_batch({k: x[k][None] for k in keys})).ravel()[0]
            rec["p_tornet"] = float(1.0 / (1.0 + np.exp(-logit)))
            rows.append(rec)
            print(f"{rec['file']} f{fr} p={rec['p_tornet']:.3f} " +
                  " ".join(f"{n}:{rec[n]['match']:.4f}/{rec[n]['prob']:.3f}" for n in dealiasers), flush=True)

    good = [r for r in rows if r["unet_vda"]["match"] >= 0.98]
    aliased = [r for r in good if r["none"]["match"] < 0.999]
    print(f"\n{len(good)} of {len(rows)} frames reproduced; {len(aliased)} have aliased gates in the chip")
    print("dealiaser      VEL match  (aliased frames, min)  fold errors  mean |dp|  max |dp|  frames |dp|>0.05  s/sweep")
    for n in dealiasers:
        m = [r[n]["match"] for r in good]
        ma = [r[n]["match"] for r in aliased] or [1.0]
        dp = np.array([abs(r[n]["prob"] - r["p_tornet"]) for r in good])
        sec = f"{np.median(times[n]):.2f}" if times[n] else "-"
        print(f"{n:13s} {np.mean(m):9.4f}  ({np.mean(ma):.4f}, {np.min(ma):.4f})"
              f"  {100 * np.mean([r[n]['fold_err'] for r in good]):9.2f}%  {dp.mean():9.4f}  {dp.max():8.3f}"
              f"  {int((dp > 0.05).sum()):16d}  {sec:>7s}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"rows": rows, "seconds_per_sweep": times}, f, indent=1)


if __name__ == "__main__":
    main()
