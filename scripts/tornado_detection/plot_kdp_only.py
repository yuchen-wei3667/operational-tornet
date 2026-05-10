import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xradar as xd

from tornet.data.kdp import (
    KDP_SOURCE_PYART_VULPIANI,
    VALID_KDP_SOURCES,
    build_target_axis,
    build_kdp_tilt_stack,
    build_tornet_tilt_sweep_pairs,
    compute_kdp_comparison_metrics,
    get_kdp_match_failures,
    interp_with_azimuth_wrap,
    mask_kdp_for_comparison,
    require_dataset_fields,
    require_datatree_sweeps,
)
from tornet.data.loader import read_file
from tornet.display.display import get_cmap


def resolve_case(manifest_path: Path, chip_name: str):
    rows = json.loads(manifest_path.read_text())
    for row in rows:
        if row.get("chip_file") == chip_name:
            return row
    raise ValueError(f"Chip file not found in manifest: {chip_name}")


def load_pyart_radar(raw_path: Path):
    try:
        import pyart
    except ImportError as exc:
        raise RuntimeError(
            "pyart is required for --kdp-source=pyart_vulpiani. "
            "Install pyart or use --kdp-source=phidp_gradient."
        ) from exc
    return pyart.io.read_nexrad_archive(str(raw_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip-file", required=True)
    parser.add_argument("--raw-file", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--chip-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tilt-index", type=int, default=0)
    parser.add_argument("--kdp-source", choices=VALID_KDP_SOURCES, default=KDP_SOURCE_PYART_VULPIANI)
    parser.add_argument("--kdp-interp-method", default="nearest")
    parser.add_argument("--kdp-windsize", type=int, default=9)
    parser.add_argument("--kdp-prefilter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kdp-phidp-interp-method", default="linear")
    parser.add_argument("--kdp-phidp-smooth-window", type=int, default=9)
    parser.add_argument("--kdp-min-abs", type=float, default=0.05)
    parser.add_argument("--kdp-min-dbz", type=float, default=20.0)
    parser.add_argument("--kdp-min-rhohv", type=float, default=0.7)
    parser.add_argument("--min-iou", type=float, default=0.20)
    parser.add_argument("--max-mad", type=float, default=0.75)
    parser.add_argument("--max-rmse", type=float, default=1.25)
    parser.add_argument("--max-centroid-offset-px", type=float, default=20.0)
    parser.add_argument("--grid-center-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kdp-plot-mode", choices=["masked", "raw"], default="masked")
    parser.add_argument("--enforce-thresholds", action="store_true")
    args = parser.parse_args()

    chip_path = Path(args.chip_root) / args.chip_file
    if args.raw_file:
        raw_path = Path(args.raw_file)
    elif args.manifest:
        raw_path = Path(resolve_case(Path(args.manifest), args.chip_file)["local_file"])
    else:
        raise ValueError("Provide either --raw-file or --manifest")

    chip = read_file(str(chip_path), n_frames=1, tilt_last=True)
    n_tilts = int(chip["KDP"].shape[-1])
    if args.tilt_index < 0 or args.tilt_index >= n_tilts:
        raise ValueError(f"tilt-index must be in [0, {n_tilts - 1}], got {args.tilt_index}.")

    chip_kdp = chip["KDP"][0, :, :, args.tilt_index].astype(np.float32)
    az = build_target_axis(
        float(chip["az_lower"][0]),
        float(chip["az_upper"][0]),
        chip["DBZ"].shape[1],
        center_sampling=args.grid_center_sampling,
    )
    rg = build_target_axis(
        float(chip["rng_lower"][0]),
        float(chip["rng_upper"][0]),
        chip["DBZ"].shape[2],
        center_sampling=args.grid_center_sampling,
    )

    dt = xd.io.open_nexradlevel2_datatree(str(raw_path))
    sweep_pairs = build_tornet_tilt_sweep_pairs(n_tilts)
    required_sweeps = sorted({idx for refl, vel in sweep_pairs for idx in (refl, vel)})
    require_datatree_sweeps(dt, required_sweeps, context="plot_kdp_only")

    radar = load_pyart_radar(raw_path) if args.kdp_source == KDP_SOURCE_PYART_VULPIANI else None
    raw_kdp_stack = build_kdp_tilt_stack(
        dt,
        az_target=az,
        range_target=rg,
        n_tilts=n_tilts,
        kdp_source=args.kdp_source,
        radar=radar,
        kdp_interp_method=args.kdp_interp_method,
        phidp_interp_method=args.kdp_phidp_interp_method,
        phidp_smooth_window=args.kdp_phidp_smooth_window,
        pyart_kdp_windsize=args.kdp_windsize,
        pyart_kdp_prefilter=args.kdp_prefilter,
        context="plot_kdp_only",
    )
    raw_kdp = raw_kdp_stack[:, :, args.tilt_index]

    refl_sweep = sweep_pairs[args.tilt_index][0]
    refl_ds = dt[f"sweep_{refl_sweep}"].to_dataset()
    require_dataset_fields(refl_ds, ["DBZH", "RHOHV"], context=f"plot_kdp_only:sweep_{refl_sweep}")
    refl_interp = interp_with_azimuth_wrap(refl_ds, az, rg, interp_method="linear")
    raw_dbz = refl_interp["DBZH"].values.astype(np.float32)
    raw_rhohv = refl_interp["RHOHV"].values.astype(np.float32)

    chip_kdp_cmp, raw_kdp_cmp, _ = mask_kdp_for_comparison(
        chip_kdp,
        raw_kdp,
        dbz=raw_dbz,
        rhohv=raw_rhohv,
        min_abs_kdp=args.kdp_min_abs,
        min_dbz=args.kdp_min_dbz,
        min_rhohv=args.kdp_min_rhohv,
    )
    metrics = compute_kdp_comparison_metrics(chip_kdp_cmp, raw_kdp_cmp, active_threshold=args.kdp_min_abs)
    failures = get_kdp_match_failures(
        metrics,
        min_iou=args.min_iou,
        max_mad=args.max_mad,
        max_rmse=args.max_rmse,
        max_centroid_offset_px=args.max_centroid_offset_px,
    )

    azr = np.deg2rad(az)
    rr = rg / 1000.0
    tt, rr2 = np.meshgrid(azr, rr, indexing="ij")
    r0 = float(chip["rng_lower"][0]) / 1000.0
    cmap, norm = get_cmap("KDP")

    fig = plt.figure(figsize=(6, 8), facecolor="0.92")
    ax1 = fig.add_subplot(2, 1, 1, polar=True)
    chip_plot = chip_kdp if args.kdp_plot_mode == "raw" else chip_kdp_cmp
    raw_plot = raw_kdp if args.kdp_plot_mode == "raw" else raw_kdp_cmp
    ax1.pcolormesh(tt, rr2 - r0, chip_plot, shading="nearest", cmap=cmap, norm=norm)
    ax1.set_theta_zero_location("N")
    ax1.set_theta_direction(-1)
    ax1.set_rorigin(-r0)
    ax1.grid(False)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_title("KDP chip (shared mask)")

    ax2 = fig.add_subplot(2, 1, 2, polar=True)
    ax2.pcolormesh(tt, rr2 - r0, raw_plot, shading="nearest", cmap=cmap, norm=norm)
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    ax2.set_rorigin(-r0)
    ax2.grid(False)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_title(
        f"KDP raw (source={args.kdp_source}, mode={args.kdp_plot_mode}, tilt={args.tilt_index}, "
        f"iou={metrics['iou']:.3f}, mad={metrics['mad']:.3f}, rmse={metrics['rmse']:.3f})"
    )

    fig.text(0.5, 0.01, f"{chip_path.name} vs {raw_path.name}", ha="center")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(args.output, dpi=150)
    plt.close(fig)

    result = {
        "chip_file": chip_path.name,
        "raw_file": raw_path.name,
        "kdp_source": args.kdp_source,
        "tilt_index": args.tilt_index,
        "kdp_plot_mode": args.kdp_plot_mode,
        "center_sampling": args.grid_center_sampling,
        "metrics": metrics,
        "threshold_failures": failures,
        "pass": len(failures) == 0,
    }
    print(json.dumps(result, sort_keys=True))
    print(f"SAVED {args.output}")

    if args.enforce_thresholds and failures:
        raise RuntimeError("KDP thresholds failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
