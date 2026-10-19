"""
Real-time TorNet inference from live NEXRAD Level-II data.

For each site, this polls the real-time Level-II chunk feed, and as soon as new
0.5/0.9 deg cuts are complete it
  1. converts them to TorNet inputs (exact TorNet conventions; U-Net VDA dealiasing;
     Level-III KDP when available, otherwise the Level-II KDP estimate),
  2. tiles the sweep with TorNet-sized chips and runs the model,
  3. writes per-chip probabilities (JSON) and a full-sweep probability map (.npz).

Example:
  python scripts/nexrad/run_realtime_tornet.py --sites KTLX KFWS --out-dir ./tornet_rt
  python scripts/nexrad/run_realtime_tornet.py --level2-file KTLX20240507_000047_V06 --out-dir .
"""
import argparse
import datetime
import json
import logging
import os
import time

import numpy as np

from tornet.nexrad import fetch
from tornet.nexrad import level2 as l2
from tornet.nexrad.dealias import UNetVDA
from tornet.nexrad.inference import SweepPredictor, load_tornet_model
from tornet.nexrad.pipeline import level2_to_tornet_volume

log = logging.getLogger("tornet_rt")


def level3_fetcher(wait_s: float, poll_s: float = 15.0):
    """Level-III KDP fetcher that waits up to wait_s for the product to be published."""
    def fetcher(site, product, cut_time):
        deadline = time.time() + wait_s
        while True:
            data = fetch.level3_kdp_fetcher(site, product, cut_time)
            if data is not None or time.time() >= deadline:
                return data
            time.sleep(poll_s)
    return fetcher


def process(volumes, predictor, dealiaser, args, tag):
    tv = level2_to_tornet_volume(volumes, level3_kdp=level3_fetcher(args.level3_wait),
                                 kdp_source=args.kdp_source, dealiaser=dealiaser, selection="latest")
    dets, prob_map = predictor.predict(tv, az_step=args.az_step, rng_step=args.rng_step * 1000.0,
                                       rng_max=args.max_range * 1000.0)
    dets = sorted(dets, key=lambda d: -d["prob"])
    stamp = tv.time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(args.out_dir, f"{tv.site or tag}_{stamp}")
    meta = {"site": tv.site, "time": tv.time.isoformat(), "vcp": tv.vcp,
            "kdp_source": tv.kdp_source, "elevations": tv.elevations.tolist(),
            "sweep_times": [t.isoformat() for t in tv.sweep_times],
            "max_prob": dets[0]["prob"] if dets else None,
            "detections": [d for d in dets if d["prob"] >= args.threshold]}
    with open(base + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    np.savez_compressed(base + "_prob.npz", prob=prob_map.astype(np.float16),
                        azimuth_deg=(np.arange(720) + 0.5) * 0.5,
                        range_m=l2.FIRST_GATE_M + l2.GATE_M * np.arange(prob_map.shape[1]))
    log.info("%s %s kdp=%s max_prob=%.3f n_above=%d -> %s", tv.site, stamp, tv.kdp_source,
             meta["max_prob"] or 0, len(meta["detections"]), base)
    return tv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="*", default=[])
    ap.add_argument("--level2-file", nargs="*", default=None, help="process local file(s) once instead")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--dealias-weights", default=None)
    ap.add_argument("--kdp-source", choices=["auto", "level3", "level2"], default="auto")
    ap.add_argument("--level3-wait", type=float, default=90.0,
                    help="seconds to wait for Level-III KDP before falling back to Level-II estimate")
    ap.add_argument("--az-step", type=float, default=15.0)
    ap.add_argument("--rng-step", type=float, default=15.0, help="km")
    ap.add_argument("--max-range", type=float, default=230.0, help="km")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--poll", type=float, default=20.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    os.makedirs(args.out_dir, exist_ok=True)

    predictor = SweepPredictor(load_tornet_model(args.model_path))
    dealiaser = UNetVDA.from_pretrained(args.dealias_weights)

    if args.level2_file:
        process(args.level2_file, predictor, dealiaser, args, "local")
        return

    # per site: current volume prefix, the previous (complete) volume, last processed cuts
    state = {s: {"prefix": None, "prev": None, "done": None} for s in args.sites}
    while True:
        for site in args.sites:
            st = state[site]
            try:
                prefix = fetch.latest_realtime_volume_prefix(site)
                if prefix != st["prefix"]:
                    # the previous volume (numbers wrap 999 -> 1) is complete now
                    num = int(prefix.rstrip("/").split("/")[-1])
                    prev_prefix = st["prefix"] or f"{site.upper()}/{num - 1 if num > 1 else 999}/"
                    try:
                        st["prev"] = l2.open_level2(fetch.realtime_volume_bytes(site, prev_prefix)[0])
                    except Exception:
                        st["prev"] = None
                    st["prefix"] = prefix
                cur = l2.open_level2(fetch.realtime_volume_bytes(site, prefix)[0])
                vols = [v for v in (st["prev"], cur) if v is not None]
                # Latest complete low-level cuts across previous + current volume; the
                # previous volume covers cuts the current one has not delivered yet.
                key = tuple(sorted(s.time for v in vols for s in v.sweeps
                                   if s.complete and s.elevation < 1.1))[-4:]
                if key and key != st["done"]:
                    process(vols, predictor, dealiaser, args, site)
                    st["done"] = key
            except Exception:
                log.exception("site %s failed", site)
        if args.once:
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
