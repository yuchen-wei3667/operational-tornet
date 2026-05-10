import argparse
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import xradar as xd
from huggingface_hub import hf_hub_download

from tornet.data import preprocess as pp
from tornet.data.kdp import (
    KDP_SOURCE_PYART_VULPIANI,
    build_kdp_tilt_stack,
    build_target_axis,
    build_tornet_tilt_sweep_pairs,
    interp_with_azimuth_wrap,
    mask_kdp_for_comparison,
    require_dataset_fields,
    require_datatree_sweeps,
)
from tornet.data.loader import read_file
from tornet.data.xradar_loader import read_nexrad_v06_to_tornet
from tornet.display.display import get_cmap
from tornet.models.keras.layers import CoordConv2D, FillNaNs


FIELDS = ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def resize_2d(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w = arr.shape
    if (in_h, in_w) == (out_h, out_w):
        return arr.astype(np.float32)

    x_in = np.linspace(0.0, 1.0, in_w)
    x_out = np.linspace(0.0, 1.0, out_w)
    tmp = np.empty((in_h, out_w), dtype=np.float32)
    for i in range(in_h):
        tmp[i] = np.interp(x_out, x_in, arr[i])

    y_in = np.linspace(0.0, 1.0, in_h)
    y_out = np.linspace(0.0, 1.0, out_h)
    out = np.empty((out_h, out_w), dtype=np.float32)
    for j in range(out_w):
        out[:, j] = np.interp(y_out, y_in, tmp[:, j])
    return out


def infer_prob_heatmap(model, heatmap_model, sample):
    xin = {k: sample[k] for k in model.input.keys()}
    hm_logit = heatmap_model.predict(xin, verbose=0)[0, :, :, 0]
    hm = sigmoid(hm_logit)
    p = float(sigmoid(float(model.predict(xin, verbose=0)[0, 0])))
    return hm, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip-file", required=True, help="Chip .nc file path")
    parser.add_argument("--raw-file", required=True, help="Raw fullscan file path")
    parser.add_argument("--output-allvars", required=True, help="Output PNG for all-vars + heatmaps")
    parser.add_argument("--output-kdp-debug", required=True, help="Output PNG for KDP debug + heatmaps")
    parser.add_argument("--tilt-index", type=int, default=0)
    parser.add_argument("--kdp-interp-method", default="nearest")
    parser.add_argument("--kdp-windsize", type=int, default=9)
    parser.add_argument("--kdp-prefilter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-path", default=None, help="Optional .keras model path")
    args = parser.parse_args()

    chip_path = Path(args.chip_file)
    raw_path = Path(args.raw_file)
    out_allvars = Path(args.output_allvars)
    out_kdp_debug = Path(args.output_kdp_debug)
    out_allvars.parent.mkdir(parents=True, exist_ok=True)
    out_kdp_debug.parent.mkdir(parents=True, exist_ok=True)

    model_path = args.model_path
    if model_path is None:
        model_path = hf_hub_download(
            repo_id="tornet-ml/tornado_detector_baseline_v1",
            filename="tornado_detector_baseline.keras",
        )

    model = keras.saving.load_model(
        model_path,
        compile=False,
        custom_objects={"CoordConv2D": CoordConv2D, "FillNaNs": FillNaNs},
    )
    heatmap_model = keras.Model(inputs=model.inputs, outputs=model.get_layer("heatmap").output)

    chip = read_file(str(chip_path), n_frames=1, tilt_last=True)
    pp.remove_time_dim(chip)
    pp.add_coordinates(chip, include_az=False, tilt_last=True, backend=np)
    pp.add_batch_dim(chip)

    raw = read_nexrad_v06_to_tornet(
        str(raw_path),
        n_sweeps=2,
        az_lower_deg=float(chip["az_lower"][0]),
        az_upper_deg=float(chip["az_upper"][0]),
        rng_lower_m=float(chip["rng_lower"][0]),
        rng_upper_m=float(chip["rng_upper"][0]),
        kdp_interp_method=args.kdp_interp_method,
        kdp_windsize=args.kdp_windsize,
        kdp_prefilter=args.kdp_prefilter,
    )
    pp.remove_time_dim(raw)
    pp.add_coordinates(raw, include_az=False, tilt_last=True, backend=np)
    pp.add_batch_dim(raw)

    hm_chip_lo, p_chip = infer_prob_heatmap(model, heatmap_model, chip)
    hm_raw_lo, p_raw = infer_prob_heatmap(model, heatmap_model, raw)

    h = int(chip["DBZ"].shape[1])
    w = int(chip["DBZ"].shape[2])
    hm_chip = resize_2d(hm_chip_lo, h, w)
    hm_raw = resize_2d(hm_raw_lo, h, w)

    az_min = float(chip["az_lower"][0]) * np.pi / 180.0
    az_max = float(chip["az_upper"][0]) * np.pi / 180.0
    rmin = float(chip["rng_lower"][0]) / 1000.0
    rmax = float(chip["rng_upper"][0]) / 1000.0
    th = np.linspace(az_min, az_max, h)
    rr = np.linspace(rmin, rmax, w)
    RR, TT = np.meshgrid(rr, th)

    fig = plt.figure(figsize=(24, 8), facecolor="0.92")
    for i, field in enumerate(FIELDS, start=1):
        ax = fig.add_subplot(2, 7, i, polar=True)
        z = chip[field][0, :, :, 0]
        cmap, norm = get_cmap(field)
        ax.pcolormesh(TT, RR - rmin, z, shading="nearest", cmap=cmap, norm=norm)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rorigin(-rmin)
        ax.set_thetalim([az_min, az_max])
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(f"Chip {field}")

    axh1 = fig.add_subplot(2, 7, 7, polar=True)
    axh1.pcolormesh(TT, RR - rmin, hm_chip, shading="nearest", cmap="inferno", vmin=0.0, vmax=1.0)
    axh1.set_theta_zero_location("N")
    axh1.set_theta_direction(-1)
    axh1.set_rorigin(-rmin)
    axh1.set_thetalim([az_min, az_max])
    axh1.grid(False)
    axh1.set_xticklabels([])
    axh1.set_yticklabels([])
    axh1.set_title("Chip heatmap prob")

    for i, field in enumerate(FIELDS, start=1):
        ax = fig.add_subplot(2, 7, 7 + i, polar=True)
        z = raw[field][0, :, :, 0]
        cmap, norm = get_cmap(field)
        ax.pcolormesh(TT, RR - rmin, z, shading="nearest", cmap=cmap, norm=norm)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rorigin(-rmin)
        ax.set_thetalim([az_min, az_max])
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(f"Raw {field}")

    axh2 = fig.add_subplot(2, 7, 14, polar=True)
    imh2 = axh2.pcolormesh(TT, RR - rmin, hm_raw, shading="nearest", cmap="inferno", vmin=0.0, vmax=1.0)
    axh2.set_theta_zero_location("N")
    axh2.set_theta_direction(-1)
    axh2.set_rorigin(-rmin)
    axh2.set_thetalim([az_min, az_max])
    axh2.grid(False)
    axh2.set_xticklabels([])
    axh2.set_yticklabels([])
    axh2.set_title("Raw heatmap prob")

    cb = fig.colorbar(imh2, ax=[axh1, axh2], location="right", shrink=0.8, pad=0.03)
    cb.set_label("heatmap probability")

    fig.suptitle(f"{chip_path.name}  vs  {raw_path.name}", y=0.98, fontsize=12)
    fig.text(0.32, 0.02, f"chip model p(tornado) = {p_chip:.6f}", ha="center", fontsize=12)
    fig.text(0.68, 0.02, f"raw model p(tornado) = {p_raw:.6f}", ha="center", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_allvars, dpi=150)
    plt.close(fig)

    # KDP masked debug + probability heatmaps
    chip0 = read_file(str(chip_path), n_frames=1, tilt_last=True)
    tilt_index = args.tilt_index
    n_tilts = int(chip0["KDP"].shape[-1])
    if tilt_index < 0 or tilt_index >= n_tilts:
        raise ValueError(f"tilt-index must be in [0, {n_tilts - 1}], got {tilt_index}.")

    az = build_target_axis(float(chip0["az_lower"][0]), float(chip0["az_upper"][0]), chip0["DBZ"].shape[1], center_sampling=True)
    rg = build_target_axis(float(chip0["rng_lower"][0]), float(chip0["rng_upper"][0]), chip0["DBZ"].shape[2], center_sampling=True)

    dt = xd.io.open_nexradlevel2_datatree(str(raw_path))
    pairs = build_tornet_tilt_sweep_pairs(n_tilts)
    req = sorted({s for pair in pairs for s in pair})
    require_datatree_sweeps(dt, req, context="replot_kdp_debug")

    import pyart

    radar = pyart.io.read_nexrad_archive(str(raw_path))
    raw_kdp_stack = build_kdp_tilt_stack(
        dt,
        az_target=az,
        range_target=rg,
        n_tilts=n_tilts,
        kdp_source=KDP_SOURCE_PYART_VULPIANI,
        radar=radar,
        kdp_interp_method=args.kdp_interp_method,
        pyart_kdp_windsize=args.kdp_windsize,
        pyart_kdp_prefilter=args.kdp_prefilter,
        context="replot_kdp_debug",
    )

    refl_sweep, _ = pairs[tilt_index]
    refl_ds = dt[f"sweep_{refl_sweep}"].to_dataset()
    require_dataset_fields(refl_ds, ["DBZH", "RHOHV"], context="replot_kdp_debug")
    refl_interp = interp_with_azimuth_wrap(refl_ds, az, rg, interp_method="linear")
    raw_dbz = refl_interp["DBZH"].values.astype(np.float32)
    raw_rhohv = refl_interp["RHOHV"].values.astype(np.float32)

    chip_kdp = chip0["KDP"][0, :, :, tilt_index].astype(np.float32)
    raw_kdp = raw_kdp_stack[:, :, tilt_index].astype(np.float32)
    chip_m, raw_m, _ = mask_kdp_for_comparison(
        chip_kdp,
        raw_kdp,
        dbz=raw_dbz,
        rhohv=raw_rhohv,
        min_abs_kdp=0.05,
        min_dbz=20.0,
        min_rhohv=0.7,
    )
    diff = raw_m - chip_m

    azr = np.deg2rad(az)
    rkm = rg / 1000.0
    tt, rr2 = np.meshgrid(azr, rkm, indexing="ij")
    r0 = float(chip0["rng_lower"][0]) / 1000.0

    kdp_cmap, kdp_norm = get_cmap("KDP")
    diff_cmap = plt.get_cmap("seismic").copy()
    diff_cmap.set_bad((0.92, 0.92, 0.92, 1.0))
    lim = float(np.nanpercentile(np.abs(diff), 99)) if np.isfinite(diff).any() else 0.1
    lim = max(lim, 0.1)

    fig2 = plt.figure(figsize=(20, 4), facecolor="0.92")
    panels = [
        ("chip_kdp_masked", chip_m, kdp_cmap, kdp_norm),
        ("raw_kdp_masked", raw_m, kdp_cmap, kdp_norm),
    ]
    for i, (title, field, cmap, norm) in enumerate(panels, start=1):
        ax = fig2.add_subplot(1, 5, i, polar=True)
        ax.pcolormesh(tt, rr2 - r0, field, shading="nearest", cmap=cmap, norm=norm)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rorigin(-r0)
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(title)

    axd = fig2.add_subplot(1, 5, 3, polar=True)
    axd.pcolormesh(tt, rr2 - r0, diff, shading="nearest", cmap=diff_cmap, vmin=-lim, vmax=lim)
    axd.set_theta_zero_location("N")
    axd.set_theta_direction(-1)
    axd.set_rorigin(-r0)
    axd.grid(False)
    axd.set_xticklabels([])
    axd.set_yticklabels([])
    axd.set_title("raw_kdp_masked - chip_kdp_masked")

    axp1 = fig2.add_subplot(1, 5, 4, polar=True)
    axp1.pcolormesh(TT, RR - rmin, hm_chip, shading="nearest", cmap="inferno", vmin=0.0, vmax=1.0)
    axp1.set_theta_zero_location("N")
    axp1.set_theta_direction(-1)
    axp1.set_rorigin(-rmin)
    axp1.set_thetalim([az_min, az_max])
    axp1.grid(False)
    axp1.set_xticklabels([])
    axp1.set_yticklabels([])
    axp1.set_title(f"chip_prob_hm p={p_chip:.3f}")

    axp2 = fig2.add_subplot(1, 5, 5, polar=True)
    imp = axp2.pcolormesh(TT, RR - rmin, hm_raw, shading="nearest", cmap="inferno", vmin=0.0, vmax=1.0)
    axp2.set_theta_zero_location("N")
    axp2.set_theta_direction(-1)
    axp2.set_rorigin(-rmin)
    axp2.set_thetalim([az_min, az_max])
    axp2.grid(False)
    axp2.set_xticklabels([])
    axp2.set_yticklabels([])
    axp2.set_title(f"raw_prob_hm p={p_raw:.3f}")

    cb2 = fig2.colorbar(imp, ax=[axp1, axp2], location="right", shrink=0.8, pad=0.03)
    cb2.set_label("probability")

    fig2.tight_layout()
    fig2.savefig(out_kdp_debug, dpi=150)
    plt.close(fig2)

    print(f"SAVED {out_allvars}")
    print(f"SAVED {out_kdp_debug}")
    print(f"p_chip={p_chip:.6f} p_raw={p_raw:.6f}")


if __name__ == "__main__":
    main()
