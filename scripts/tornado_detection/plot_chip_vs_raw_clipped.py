import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xradar as xd

from tornet.data.loader import read_file
from tornet.display.display import get_cmap


FIELD_SPECS = [
    ("DBZ", "DBZH", 0),
    ("VEL", "VRADH", 1),
    ("KDP", "PHIDP", 0),
    ("RHOHV", "RHOHV", 0),
    ("ZDR", "ZDR", 0),
    ("WIDTH", "WRADH", 1),
]


def build_kdp_from_phidp(
    ds_sweep,
    az,
    rg,
    sign=-1.0,
    az_flip=False,
    r_flip=False,
    az_roll: int = 0,
    smooth_window: int = 9,
):
    ph = ds_sweep.interp(azimuth=az, range=rg, method="linear")["PHIDP"].values.astype(np.float32)
    if smooth_window > 1:
        kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
        ph = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, ph)
    kdp = np.gradient(ph, np.maximum(rg / 1000.0, 1e-3), axis=1).astype(np.float32)
    if az_flip:
        kdp = kdp[::-1, :]
    if r_flip:
        kdp = kdp[:, ::-1]
    if az_roll:
        kdp = np.roll(kdp, az_roll, axis=0)
    kdp = sign * kdp
    return kdp


def choose_best_kdp_alignment(dt, chip_kdp, az, rg):
    best = None
    chip_mask = np.isfinite(chip_kdp) & (np.abs(chip_kdp) > 0.05)
    for sweep_idx in (0, 2):
        ds = dt[f"sweep_{sweep_idx}"].to_dataset().sortby("azimuth")
        for sign in (1.0, -1.0):
            for az_flip in (False, True):
                for r_flip in (False, True):
                    ds_interp = ds.interp(azimuth=az, range=rg, method="linear")
                    dbz = ds_interp["DBZH"].values.astype(np.float32)
                    rhohv = ds_interp["RHOHV"].values.astype(np.float32)
                    for smooth_window in (3, 5, 7, 9, 11, 15):
                        for az_roll in range(-12, 13):
                            raw_kdp = build_kdp_from_phidp(
                                ds,
                                az,
                                rg,
                                sign=sign,
                                az_flip=az_flip,
                                r_flip=r_flip,
                                az_roll=az_roll,
                                smooth_window=smooth_window,
                            )
                            if az_flip:
                                dbz_cmp = dbz[::-1, :]
                                rhohv_cmp = rhohv[::-1, :]
                            else:
                                dbz_cmp = dbz
                                rhohv_cmp = rhohv
                            if r_flip:
                                dbz_cmp = dbz_cmp[:, ::-1]
                                rhohv_cmp = rhohv_cmp[:, ::-1]
                            if az_roll:
                                dbz_cmp = np.roll(dbz_cmp, az_roll, axis=0)
                                rhohv_cmp = np.roll(rhohv_cmp, az_roll, axis=0)

                            mask = np.isfinite(chip_kdp) & np.isfinite(raw_kdp)
                            if mask.sum() < 10:
                                continue
                            qc_mask = mask & (dbz_cmp >= 20.0) & (rhohv_cmp >= 0.7)
                            raw_mask = qc_mask & (np.abs(raw_kdp) > 0.05)
                            inter = int((chip_mask & raw_mask).sum())
                            union = int((chip_mask | raw_mask).sum())
                            iou = inter / union if union else 0.0

                            chip_pts = np.argwhere(chip_mask)
                            raw_pts = np.argwhere(raw_mask)
                            if len(chip_pts) and len(raw_pts):
                                chip_ctr = chip_pts.mean(axis=0)
                                raw_ctr = raw_pts.mean(axis=0)
                                ctr_dist = float(np.linalg.norm(chip_ctr - raw_ctr))
                            else:
                                ctr_dist = float("inf")

                            focus = chip_mask | raw_mask
                            use = focus if focus.sum() >= 10 else mask
                            mad = float(np.mean(np.abs(chip_kdp[use] - raw_kdp[use])))

                            item = (
                                iou,
                                -ctr_dist,
                                -mad,
                                sweep_idx,
                                sign,
                                az_flip,
                                r_flip,
                                az_roll,
                                smooth_window,
                                raw_kdp,
                            )
                            if best is None or item[:3] > best[:3]:
                                best = item
    return best


def plot_case(chip_path: Path, raw_path: Path, out_png: Path, kdp_sign: float, kdp_az_flip: bool, auto_kdp_align: bool):
    chip = read_file(str(chip_path), n_frames=1, tilt_last=True)
    az = np.linspace(float(chip["az_lower"][0]), float(chip["az_upper"][0]), chip["DBZ"].shape[1]).astype(np.float32)
    rg = np.linspace(float(chip["rng_lower"][0]), float(chip["rng_upper"][0]), chip["DBZ"].shape[2]).astype(np.float32)

    dt = xd.io.open_nexradlevel2_datatree(str(raw_path))
    chip_kdp = chip["KDP"][0, :, :, 0].astype(np.float32)
    if auto_kdp_align:
        best = choose_best_kdp_alignment(dt, chip_kdp, az, rg)
        if best is None:
            ds0 = dt["sweep_0"].to_dataset().sortby("azimuth")
            kdp_raw = build_kdp_from_phidp(ds0, az, rg, sign=kdp_sign, az_flip=kdp_az_flip, r_flip=False, az_roll=0)
            kdp_meta = "fallback sweep0"
        else:
            _, _, _, kdp_sw, kdp_sign, kdp_az_flip, kdp_r_flip, kdp_az_roll, kdp_smooth_window, kdp_raw = best
            kdp_meta = (
                f"auto sw={kdp_sw}, sign={int(kdp_sign)}, az_flip={kdp_az_flip}, "
                f"r_flip={kdp_r_flip}, roll={kdp_az_roll}, smooth={kdp_smooth_window}"
            )
    else:
        ds0 = dt["sweep_0"].to_dataset().sortby("azimuth")
        kdp_raw = build_kdp_from_phidp(ds0, az, rg, sign=kdp_sign, az_flip=kdp_az_flip, r_flip=False, az_roll=0)
        kdp_meta = f"manual sign={int(kdp_sign)}, az_flip={kdp_az_flip}"

    fig = plt.figure(figsize=(18, 6), facecolor="0.92")

    for i, (chip_field, raw_field, sweep_idx) in enumerate(FIELD_SPECS, start=1):
        chip_plot = chip[chip_field][0, :, :, 0].astype(np.float32)

        if chip_field == "KDP":
            raw_plot = kdp_raw.copy()
            # TorNet KDP is much more aggressively filtered/QC'd than a raw
            # PHIDP gradient. Restrict the raw display to the chip's valid KDP
            # footprint so the comparison is spatially meaningful.
            chip_valid = np.isfinite(chip_plot) & (np.abs(chip_plot) > 0.05)
            raw_plot[~chip_valid] = np.nan
        else:
            ds = dt[f"sweep_{sweep_idx}"].to_dataset().sortby("azimuth")
            raw_plot = ds.interp(azimuth=az, range=rg, method="linear")[raw_field].values.astype(np.float32)

        azr = np.deg2rad(az)
        rr = rg / 1000.0
        tt, rr2 = np.meshgrid(azr, rr, indexing="ij")
        r0 = float(chip["rng_lower"][0]) / 1000.0

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
            ax2.set_title(f"KDP raw clipped ({kdp_meta})")
        else:
            ax2.set_title(f"{chip_field} raw clipped")

    fig.text(0.5, 0.01, f"{chip_path.name} vs {raw_path.name}", ha="center")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to download_manifest.json")
    parser.add_argument("--chip-root", required=True, help="Directory containing TorNet .nc chip files")
    parser.add_argument("--out-dir", required=True, help="Output directory for plots")
    parser.add_argument("--only-chip", default=None, help="Optional specific chip filename to plot")
    parser.add_argument("--kdp-sign", type=float, default=-1.0, help="Multiply KDP by this sign")
    parser.add_argument("--kdp-az-flip", action="store_true", help="Flip KDP along azimuth axis")
    parser.add_argument("--auto-kdp-align", action="store_true", help="Auto-search sweep/sign/flips for KDP best alignment")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(Path(args.manifest).read_text())
    chip_root = Path(args.chip_root)

    for rec in rows:
        if not rec.get("downloaded"):
            continue
        if args.only_chip and rec["chip_file"] != args.only_chip:
            continue

        chip_path = chip_root / rec["chip_file"]
        raw_path = Path(rec["local_file"])
        out_png = out_dir / f"{rec['class']}_{chip_path.stem}_vs_{raw_path.name}.png"

        plot_case(
            chip_path=chip_path,
            raw_path=raw_path,
            out_png=out_png,
            kdp_sign=args.kdp_sign,
            kdp_az_flip=args.kdp_az_flip,
            auto_kdp_align=args.auto_kdp_align,
        )
        print(f"SAVED {out_png}")


if __name__ == "__main__":
    main()
