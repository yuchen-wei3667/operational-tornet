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
    sanitize_velocity_sweep_dataset,
)
from tornet.data.loader import read_file
from tornet.display.display import get_cmap


FIELD_SPECS = [
    ("DBZ", "DBZH", "reflectivity"),
    ("VEL", "VRADH", "velocity"),
    ("KDP", "KDP", "kdp"),
    ("RHOHV", "RHOHV", "reflectivity"),
    ("ZDR", "ZDR", "reflectivity"),
    ("WIDTH", "WRADH", "velocity"),
]


def _percentiles(a: np.ndarray) -> list[float]:
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return [float("nan")] * 7
    return [float(v) for v in np.percentile(finite, [0, 1, 5, 50, 95, 99, 100])]


def load_pyart_radar(raw_path: Path):
    try:
        import pyart
    except ImportError as exc:
        raise RuntimeError(
            "pyart is required for --kdp-source=pyart_vulpiani. "
            "Install pyart or use --kdp-source=phidp_gradient."
        ) from exc
    return pyart.io.read_nexrad_archive(str(raw_path))


def plot_case(
    chip_path: Path,
    raw_path: Path,
    out_png: Path,
    *,
    tilt_index: int,
    kdp_source: str,
    kdp_interp_method: str,
    kdp_windsize: int,
    kdp_prefilter: bool,
    kdp_phidp_interp_method: str,
    kdp_phidp_smooth_window: int,
    kdp_min_abs: float,
    kdp_min_dbz: float,
    kdp_min_rhohv: float,
    min_iou: float,
    max_mad: float,
    max_rmse: float,
    max_centroid_offset_px: float,
    center_sampling: bool,
    kdp_plot_mode: str,
    kdp_debug_output: Path | None,
):
    chip = read_file(str(chip_path), n_frames=1, tilt_last=True)
    n_tilts = int(chip["KDP"].shape[-1])
    if tilt_index < 0 or tilt_index >= n_tilts:
        raise ValueError(f"tilt-index must be in [0, {n_tilts - 1}], got {tilt_index}.")

    az = build_target_axis(
        float(chip["az_lower"][0]),
        float(chip["az_upper"][0]),
        chip["DBZ"].shape[1],
        center_sampling=center_sampling,
    )
    rg = build_target_axis(
        float(chip["rng_lower"][0]),
        float(chip["rng_upper"][0]),
        chip["DBZ"].shape[2],
        center_sampling=center_sampling,
    )

    dt = xd.io.open_nexradlevel2_datatree(str(raw_path))
    sweep_pairs = build_tornet_tilt_sweep_pairs(n_tilts)
    required_sweeps = sorted({idx for refl, vel in sweep_pairs for idx in (refl, vel)})
    require_datatree_sweeps(dt, required_sweeps, context=f"plot_case:{chip_path.name}")

    radar = load_pyart_radar(raw_path) if kdp_source == KDP_SOURCE_PYART_VULPIANI else None
    raw_kdp_stack = build_kdp_tilt_stack(
        dt,
        az_target=az,
        range_target=rg,
        n_tilts=n_tilts,
        kdp_source=kdp_source,
        radar=radar,
        kdp_interp_method=kdp_interp_method,
        phidp_interp_method=kdp_phidp_interp_method,
        phidp_smooth_window=kdp_phidp_smooth_window,
        pyart_kdp_windsize=kdp_windsize,
        pyart_kdp_prefilter=kdp_prefilter,
        context=f"plot_case:{chip_path.name}",
    )

    refl_sweep, vel_sweep = sweep_pairs[tilt_index]
    refl_ds = dt[f"sweep_{refl_sweep}"].to_dataset()
    vel_ds = dt[f"sweep_{vel_sweep}"].to_dataset()
    require_dataset_fields(
        refl_ds,
        ["DBZH", "RHOHV", "ZDR"],
        context=f"plot_case:{chip_path.name}:sweep_{refl_sweep}",
    )
    require_dataset_fields(
        vel_ds,
        ["VRADH", "WRADH"],
        context=f"plot_case:{chip_path.name}:sweep_{vel_sweep}",
    )
    vel_ds = sanitize_velocity_sweep_dataset(vel_ds)
    refl_interp = interp_with_azimuth_wrap(refl_ds, az, rg, interp_method="linear")
    vel_interp = interp_with_azimuth_wrap(vel_ds, az, rg, interp_method="linear")

    chip_kdp = chip["KDP"][0, :, :, tilt_index].astype(np.float32)
    raw_kdp = raw_kdp_stack[:, :, tilt_index].astype(np.float32)
    kdp_diff = raw_kdp - chip_kdp
    raw_dbz = refl_interp["DBZH"].values.astype(np.float32)
    raw_rhohv = refl_interp["RHOHV"].values.astype(np.float32)

    chip_kdp_cmp, raw_kdp_cmp, _ = mask_kdp_for_comparison(
        chip_kdp,
        raw_kdp,
        dbz=raw_dbz,
        rhohv=raw_rhohv,
        min_abs_kdp=kdp_min_abs,
        min_dbz=kdp_min_dbz,
        min_rhohv=kdp_min_rhohv,
    )
    metrics = compute_kdp_comparison_metrics(chip_kdp_cmp, raw_kdp_cmp, active_threshold=kdp_min_abs)
    failures = get_kdp_match_failures(
        metrics,
        min_iou=min_iou,
        max_mad=max_mad,
        max_rmse=max_rmse,
        max_centroid_offset_px=max_centroid_offset_px,
    )

    fig = plt.figure(figsize=(18, 6), facecolor="0.92")
    azr = np.deg2rad(az)
    rr = rg / 1000.0
    tt, rr2 = np.meshgrid(azr, rr, indexing="ij")
    r0 = float(chip["rng_lower"][0]) / 1000.0

    for i, (chip_field, raw_field, source_kind) in enumerate(FIELD_SPECS, start=1):
        chip_plot = chip[chip_field][0, :, :, tilt_index].astype(np.float32)
        if source_kind == "reflectivity":
            raw_plot = refl_interp[raw_field].values.astype(np.float32)
        elif source_kind == "velocity":
            raw_plot = vel_interp[raw_field].values.astype(np.float32)
        else:
            if kdp_plot_mode == "raw":
                chip_plot = chip_kdp
                raw_plot = raw_kdp
            else:
                chip_plot = chip_kdp_cmp
                raw_plot = raw_kdp_cmp

        cmap, norm = get_cmap(chip_field)

        ax1 = fig.add_subplot(2, 6, i, polar=True)
        ax1.pcolormesh(tt, rr2 - r0, chip_plot, shading="nearest", cmap=cmap, norm=norm)
        ax1.set_theta_zero_location("N")
        ax1.set_theta_direction(-1)
        ax1.set_rorigin(-r0)
        ax1.grid(False)
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_title(f"{chip_field} chip")

        ax2 = fig.add_subplot(2, 6, 6 + i, polar=True)
        ax2.pcolormesh(tt, rr2 - r0, raw_plot, shading="nearest", cmap=cmap, norm=norm)
        ax2.set_theta_zero_location("N")
        ax2.set_theta_direction(-1)
        ax2.set_rorigin(-r0)
        ax2.grid(False)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        if chip_field == "KDP":
            ax2.set_title(
                f"KDP raw ({kdp_source}, mode={kdp_plot_mode}, iou={metrics['iou']:.3f}, "
                f"mad={metrics['mad']:.3f}, rmse={metrics['rmse']:.3f})"
            )
        else:
            ax2.set_title(f"{chip_field} raw")

    fig.text(0.5, 0.01, f"{chip_path.name} vs {raw_path.name} (tilt={tilt_index})", ha="center")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    if kdp_debug_output is not None:
        if kdp_plot_mode == "raw":
            chip_kdp_dbg = chip_kdp
            raw_kdp_dbg = raw_kdp
            chip_dbg_title = "chip_kdp_raw"
            raw_dbg_title = "raw_kdp_raw"
        else:
            chip_kdp_dbg = chip_kdp_cmp
            raw_kdp_dbg = raw_kdp_cmp
            chip_dbg_title = "chip_kdp_masked"
            raw_dbg_title = "raw_kdp_masked"
        kdp_diff_dbg = raw_kdp_dbg - chip_kdp_dbg

        diff_cmap = plt.get_cmap("seismic").copy()
        diff_cmap.set_bad((0.92, 0.92, 0.92, 1.0))
        diff_abs_p99 = float(np.nanpercentile(np.abs(kdp_diff_dbg), 99)) if np.isfinite(kdp_diff_dbg).any() else 1.0
        diff_lim = max(diff_abs_p99, 0.1)
        kdp_cmap, kdp_norm = get_cmap("KDP")

        dfig = plt.figure(figsize=(12, 4), facecolor="0.92")
        panels = [(chip_dbg_title, chip_kdp_dbg, kdp_cmap, kdp_norm), (raw_dbg_title, raw_kdp_dbg, kdp_cmap, kdp_norm)]
        for idx, (title, field, cmap, norm) in enumerate(panels, start=1):
            ax = dfig.add_subplot(1, 3, idx, polar=True)
            ax.pcolormesh(tt, rr2 - r0, field, shading="nearest", cmap=cmap, norm=norm)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_rorigin(-r0)
            ax.grid(False)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title(title)

        axd = dfig.add_subplot(1, 3, 3, polar=True)
        axd.pcolormesh(
            tt,
            rr2 - r0,
            kdp_diff_dbg,
            shading="nearest",
            cmap=diff_cmap,
            vmin=-diff_lim,
            vmax=diff_lim,
        )
        axd.set_theta_zero_location("N")
        axd.set_theta_direction(-1)
        axd.set_rorigin(-r0)
        axd.grid(False)
        axd.set_xticklabels([])
        axd.set_yticklabels([])
        axd.set_title(f"{raw_dbg_title} - {chip_dbg_title}")

        dfig.tight_layout()
        dfig.savefig(kdp_debug_output, dpi=150)
        plt.close(dfig)

    return {
        "chip_file": chip_path.name,
        "raw_file": raw_path.name,
        "tilt_index": tilt_index,
        "kdp_source": kdp_source,
        "kdp_plot_mode": kdp_plot_mode,
        "center_sampling": center_sampling,
        "metrics": metrics,
        "chip_kdp_percentiles": _percentiles(chip_kdp),
        "raw_kdp_percentiles": _percentiles(raw_kdp),
        "diff_kdp_percentiles": _percentiles(kdp_diff),
        "threshold_failures": failures,
        "pass": len(failures) == 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to download_manifest.json")
    parser.add_argument("--chip-root", required=True, help="Directory containing TorNet .nc chip files")
    parser.add_argument("--out-dir", required=True, help="Output directory for plots")
    parser.add_argument("--only-chip", default=None, help="Optional specific chip filename to plot")
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
    parser.add_argument(
        "--kdp-debug-output",
        default=None,
        help="Optional path for an unmasked chip/raw/diff KDP debug figure.",
    )
    parser.add_argument("--enforce-thresholds", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(Path(args.manifest).read_text())
    chip_root = Path(args.chip_root)

    failures: list[dict] = []
    for rec in rows:
        if not rec.get("downloaded"):
            continue
        if args.only_chip and rec["chip_file"] != args.only_chip:
            continue

        chip_path = chip_root / rec["chip_file"]
        raw_path = Path(rec["local_file"])
        out_png = out_dir / f"{rec['class']}_{chip_path.stem}_vs_{raw_path.name}.png"

        result = plot_case(
            chip_path=chip_path,
            raw_path=raw_path,
            out_png=out_png,
            tilt_index=args.tilt_index,
            kdp_source=args.kdp_source,
            kdp_interp_method=args.kdp_interp_method,
            kdp_windsize=args.kdp_windsize,
            kdp_prefilter=args.kdp_prefilter,
            kdp_phidp_interp_method=args.kdp_phidp_interp_method,
            kdp_phidp_smooth_window=args.kdp_phidp_smooth_window,
            kdp_min_abs=args.kdp_min_abs,
            kdp_min_dbz=args.kdp_min_dbz,
            kdp_min_rhohv=args.kdp_min_rhohv,
            min_iou=args.min_iou,
            max_mad=args.max_mad,
            max_rmse=args.max_rmse,
            max_centroid_offset_px=args.max_centroid_offset_px,
            center_sampling=args.grid_center_sampling,
            kdp_plot_mode=args.kdp_plot_mode,
            kdp_debug_output=Path(args.kdp_debug_output) if args.kdp_debug_output else None,
        )
        print(json.dumps(result, sort_keys=True))
        print(f"SAVED {out_png}")
        if not result["pass"]:
            failures.append(result)

    if args.enforce_thresholds and failures:
        failed_cases = ", ".join(item["chip_file"] for item in failures)
        raise RuntimeError(f"KDP thresholds failed for {len(failures)} cases: {failed_cases}")


if __name__ == "__main__":
    main()
