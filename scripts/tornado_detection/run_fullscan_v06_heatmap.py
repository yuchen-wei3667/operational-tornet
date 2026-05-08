import argparse
import os

import keras
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from huggingface_hub import hf_hub_download

from tornet.data import preprocess as pp
from tornet.data.xradar_loader import read_nexrad_v06_to_tornet
from tornet.display.display import get_cmap, get_label
from tornet.models.keras.layers import CoordConv2D, FillNaNs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_path", help="Path to NEXRAD V06 full scan file")
    parser.add_argument("--model_path", default=None, help="Path to .keras model")
    parser.add_argument(
        "--output",
        default="fullscan_v06_tornet_heatmap.png",
        help="Output figure path",
    )
    parser.add_argument("--heatmap_vmin", type=float, default=None, help="Optional fixed heatmap colorbar min")
    parser.add_argument("--heatmap_vmax", type=float, default=None, help="Optional fixed heatmap colorbar max")
    parser.add_argument(
        "--heatmap_as_probability",
        action="store_true",
        help="Plot sigmoid(heatmap) so values are in [0, 1]",
    )
    args = parser.parse_args()

    trained_model = args.model_path
    if trained_model is None:
        trained_model = hf_hub_download(
            repo_id="tornet-ml/tornado_detector_baseline_v1",
            filename="tornado_detector_baseline.keras",
        )

    model = keras.saving.load_model(
        trained_model,
        compile=False,
        custom_objects={"CoordConv2D": CoordConv2D, "FillNaNs": FillNaNs},
    )

    # Build TorNet-style input from xradar-opened full scan
    sample = read_nexrad_v06_to_tornet(args.scan_path, n_sweeps=2)
    pp.remove_time_dim(sample)
    pp.add_coordinates(sample, include_az=False, tilt_last=True, backend=np)
    pp.add_batch_dim(sample)

    xin = {k: sample[k] for k in model.input.keys()}

    # "Remove" final GlobalMaxPooling layer by probing named pre-pool layer.
    heatmap_model = keras.Model(inputs=model.inputs, outputs=model.get_layer("heatmap").output)
    heatmap = heatmap_model.predict(xin, verbose=0)[0, :, :, 0]
    if args.heatmap_as_probability:
        heatmap = 1.0 / (1.0 + np.exp(-heatmap))
    logit = model.predict(xin, verbose=0)
    prob = 1.0 / (1.0 + np.exp(-logit))[0, 0]

    az_min = float(sample["az_lower"][0]) * np.pi / 180.0
    az_max = float(sample["az_upper"][0]) * np.pi / 180.0
    rmin_km = float(sample["rng_lower"][0]) / 1000.0
    rmax_km = float(sample["rng_upper"][0]) / 1000.0

    na = sample["DBZ"].shape[1]
    nr = sample["DBZ"].shape[2]
    theta = np.linspace(az_min, az_max, na)
    rr = np.linspace(rmin_km, rmax_km, nr)
    RR, TT = np.meshgrid(rr, theta)

    hna, hnr = heatmap.shape
    htheta = np.linspace(az_min, az_max, hna)
    hrr = np.linspace(rmin_km, rmax_km, hnr)
    hRR, hTT = np.meshgrid(hrr, htheta)

    fields = ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH"]
    fig = plt.figure(figsize=(16, 8), facecolor="0.9")

    for i, field in enumerate(fields, start=1):
        ax = fig.add_subplot(2, 4, i, polar=True)
        z = sample[field][0, :, :, 0]
        cmap, norm = get_cmap(field)
        im = ax.pcolormesh(TT, RR - rmin_km, z, shading="nearest", cmap=cmap, norm=norm)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rorigin(-rmin_km)
        ax.set_thetalim([az_min, az_max])
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(field)
        cbar = fig.colorbar(im, ax=ax, location="right", shrink=0.65, pad=0.05)
        cbar.set_label(get_label(field))

    axh = fig.add_subplot(2, 4, 7, polar=True)
    imh = axh.pcolormesh(hTT, hRR - rmin_km, heatmap, shading="nearest", cmap="inferno")
    axh.set_theta_zero_location("N")
    axh.set_theta_direction(-1)
    axh.set_rorigin(-rmin_km)
    axh.set_thetalim([az_min, az_max])
    axh.grid(False)
    axh.set_xticklabels([])
    axh.set_yticklabels([])
    if args.heatmap_as_probability:
        axh.set_title("HEATMAP PROBABILITY (sigmoid pre-GMP)")
        if args.heatmap_vmin is not None or args.heatmap_vmax is not None:
            vmin = args.heatmap_vmin if args.heatmap_vmin is not None else 0.0
            vmax = args.heatmap_vmax if args.heatmap_vmax is not None else 1.0
        else:
            vmin, vmax = 0.0, 1.0
        imh.set_norm(mpl.colors.Normalize(vmin=vmin, vmax=vmax))
    else:
        axh.set_title("HEATMAP (pre-GMP)")
        if args.heatmap_vmin is not None or args.heatmap_vmax is not None:
            vmin = args.heatmap_vmin if args.heatmap_vmin is not None else float(np.nanmin(heatmap))
            vmax = args.heatmap_vmax if args.heatmap_vmax is not None else float(np.nanmax(heatmap))
            imh.set_norm(mpl.colors.Normalize(vmin=vmin, vmax=vmax))
    cbarh = fig.colorbar(imh, ax=axh, location="right", shrink=0.65, pad=0.05)
    cbarh.set_label("probability" if args.heatmap_as_probability else "activation")

    fig.add_subplot(2, 4, 8).axis("off")

    fig.text(
        0.5,
        0.03,
        f"{os.path.basename(args.scan_path)}  p(tornado)={prob:.3f}",
        ha="center",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(args.output, dpi=180)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
