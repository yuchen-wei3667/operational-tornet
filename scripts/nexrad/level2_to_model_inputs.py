"""
Convert NEXRAD Level-II scans into model-ready TorNet inputs (.npz).

Each output file holds float32 arrays [N, 120, 240, 2] named like the model inputs
(DBZ, VEL, KDP, RHOHV, ZDR, WIDTH, range_folded_mask, coordinates) plus chip_* and
meta_* entries; load with ``tornet.nexrad.model_inputs.ModelInputs.load`` and pass
``.inputs`` straight to ``model.predict``.

Examples:
  # local file, whole sweep tiled, KDP from Level-III on AWS (Level-II estimate if missing)
  python scripts/nexrad/level2_to_model_inputs.py KTLX20240507_000047_V06 --out-dir inputs/
  # chosen chips only, offline (KDP estimated from Level-II)
  python scripts/nexrad/level2_to_model_inputs.py KTLX20240507_000047_V06 --no-level3 \
      --center 215 74 --center 230 90
  # newest live scan from the real-time feed
  python scripts/nexrad/level2_to_model_inputs.py --site KTLX --out-dir inputs/
"""
import argparse
import logging
import os
import time

from tornet.nexrad import fetch
from tornet.nexrad.model_inputs import level2_to_model_inputs

log = logging.getLogger("level2_to_model_inputs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("level2_files", nargs="*", help="Level-II files; each is converted separately")
    ap.add_argument("--site", nargs="*", default=[], help="convert the newest real-time scan of these sites")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--center", nargs=2, type=float, action="append", metavar=("AZ_DEG", "RNG_KM"),
                    help="chip center (repeatable); default tiles the whole sweep")
    ap.add_argument("--az-step", type=float, default=15.0)
    ap.add_argument("--rng-step", type=float, default=15.0, help="km")
    ap.add_argument("--max-range", type=float, default=230.0, help="km")
    ap.add_argument("--kdp-source", choices=["auto", "level3", "level2"], default="auto")
    ap.add_argument("--no-level3", action="store_true", help="do not fetch Level-III KDP (offline)")
    ap.add_argument("--no-dealias", action="store_true")
    ap.add_argument("--include-az", action="store_true", help="add azimuth to coordinates (models trained with it)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not args.level2_files and not args.site:
        ap.error("give Level-II files and/or --site")
    os.makedirs(args.out_dir, exist_ok=True)

    sources = [(os.path.basename(f), f) for f in args.level2_files]
    for site in args.site:
        data, complete = fetch.realtime_volume_bytes(site)
        log.info("%s: newest real-time volume (%s)", site, "complete" if complete else "in progress")
        sources.append((site.upper(), data))

    centers = [(a, r * 1000.0) for a, r in args.center] if args.center else None
    for name, src in sources:
        t0 = time.time()
        mi = level2_to_model_inputs(
            src, centers=centers,
            level3_kdp=None if args.no_level3 else fetch.level3_kdp_fetcher,
            kdp_source=args.kdp_source, dealiaser=None if args.no_dealias else "default",
            include_az=args.include_az,
            az_step=args.az_step, rng_step=args.rng_step * 1000.0, rng_max=args.max_range * 1000.0)
        stamp = mi.time.strftime("%Y%m%d_%H%M%S") if mi.time else "unknown"
        out = os.path.join(args.out_dir, f"{mi.site or name}_{stamp}_inputs.npz")
        mi.save(out)
        log.info("%s -> %s: %d chips, kdp=%s, %.1fs", name, out, len(mi), mi.kdp_source, time.time() - t0)


if __name__ == "__main__":
    main()
