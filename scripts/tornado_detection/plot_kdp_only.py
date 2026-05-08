import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xradar as xd
import xarray as xr

from tornet.data.loader import read_file
from tornet.display.display import get_cmap


def _import_pyart():
    try:
        import pyart  # type: ignore

        return pyart
    except ImportError:
        return None


def _pyart_sweep_field(radar, field_data: np.ndarray, sweep_idx: int):
    s0 = int(radar.sweep_start_ray_index["data"][sweep_idx])
    s1 = int(radar.sweep_end_ray_index["data"][sweep_idx]) + 1
    az = np.asarray(radar.azimuth["data"][s0:s1], dtype=np.float32)
    rg = np.asarray(radar.range["data"], dtype=np.float32)
    fld = np.asarray(field_data[s0:s1, :], dtype=np.float32)
    order = np.argsort(az)
    return az[order], rg, fld[order]


def build_kdp_from_pyart(raw_path: Path, sweep_idx: int, az, rg, interp_method: str = "linear"):
    pyart = _import_pyart()
    if pyart is None:
        return None

    radar = pyart.io.read_nexrad_archive(str(raw_path))
    kdp_dict, _ = pyart.retrieve.kdp_vulpiani(
        radar,
        phidp_field="differential_phase",
        band="S",
        windsize=20,
        prefilter_psidp=True,
    )
    kdp_data = np.asarray(kdp_dict["data"], dtype=np.float32)
    kaz, krg, kfld = _pyart_sweep_field(radar, kdp_data, sweep_idx)
    kds = xr.Dataset({"kdp": (("azimuth", "range"), kfld)}, coords={"azimuth": kaz, "range": krg})
    return kds.interp(
        azimuth=xr.DataArray(az, dims=["azimuth"]),
        range=xr.DataArray(rg, dims=["range"]),
        method=interp_method,
    )["kdp"].values.astype(np.float32)


def get_lowest_kdp_sweeps(dt, max_count: int = 2):
    sweep_ids = []
    for key in dt.keys():
        if not str(key).startswith("sweep_"):
            continue
        try:
            idx = int(str(key).split("_")[1])
        except (IndexError, ValueError):
            continue
        ds = dt[str(key)].to_dataset()
        if "PHIDP" in ds:
            sweep_ids.append(idx)
    sweep_ids.sort()
    if not sweep_ids:
        raise ValueError("No KDP-capable sweeps found (missing PHIDP).")
    return sweep_ids[:max_count]


def build_kdp_from_phidp(
    ds_sweep,
    az,
    rg,
    sign=-1.0,
    az_flip=False,
    r_flip=False,
    az_roll=0,
    smooth_window=9,
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
    return sign * kdp


def choose_best_kdp_alignment(dt, chip_kdp, az, rg, sweeps, raw_kdp_by_sweep=None):
    best = None
    chip_mask = np.isfinite(chip_kdp) & (np.abs(chip_kdp) > 0.05)
    for sweep_idx in sweeps:
        ds = dt[f"sweep_{sweep_idx}"].to_dataset().sortby("azimuth")
        ds_interp = ds.interp(azimuth=az, range=rg, method="linear")
        dbz = ds_interp["DBZH"].values.astype(np.float32)
        rhohv = ds_interp["RHOHV"].values.astype(np.float32)
        for sign in (1.0, -1.0):
            for az_flip in (False, True):
                for r_flip in (False, True):
                    for smooth_window in (3, 5, 7, 9, 11, 15):
                        for az_roll in range(-12, 13):
                            if raw_kdp_by_sweep is None:
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
                            else:
                                raw_kdp = sign * raw_kdp_by_sweep[sweep_idx]
                                if az_flip:
                                    raw_kdp = raw_kdp[::-1, :]
                                if r_flip:
                                    raw_kdp = raw_kdp[:, ::-1]
                                if az_roll:
                                    raw_kdp = np.roll(raw_kdp, az_roll, axis=0)
                            dbz_cmp = dbz[::-1, :] if az_flip else dbz
                            rhohv_cmp = rhohv[::-1, :] if az_flip else rhohv
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

                            item = {
                                "iou": iou,
                                "ctr_dist": ctr_dist,
                                "mad": mad,
                                "sweep": sweep_idx,
                                "sign": sign,
                                "az_flip": az_flip,
                                "r_flip": r_flip,
                                "az_roll": az_roll,
                                "smooth_window": smooth_window,
                                "raw_kdp": raw_kdp,
                            }
                            if best is None or (item["iou"], -item["ctr_dist"], -item["mad"]) > (
                                best["iou"],
                                -best["ctr_dist"],
                                -best["mad"],
                            ):
                                best = item
    return best


def resolve_case(manifest_path: Path, chip_name: str):
    rows = json.loads(manifest_path.read_text())
    for row in rows:
        if row.get("chip_file") == chip_name:
            return row
    raise ValueError(f"Chip file not found in manifest: {chip_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip-file", required=True)
    parser.add_argument("--raw-file", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--chip-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--auto-align", action="store_true")
    parser.add_argument("--kdp-sign", type=float, default=-1.0)
    parser.add_argument("--kdp-az-flip", action="store_true")
    parser.add_argument("--kdp-r-flip", action="store_true")
    parser.add_argument("--kdp-az-roll", type=int, default=0)
    parser.add_argument("--kdp-smooth-window", type=int, default=9)
    args = parser.parse_args()

    chip_path = Path(args.chip_root) / args.chip_file
    if args.raw_file:
        raw_path = Path(args.raw_file)
    elif args.manifest:
        raw_path = Path(resolve_case(Path(args.manifest), args.chip_file)["local_file"])
    else:
        raise ValueError("Provide either --raw-file or --manifest")

    chip = read_file(str(chip_path), n_frames=1, tilt_last=True)
    chip_kdp = chip["KDP"][0, :, :, 0].astype(np.float32)
    chip_valid = np.isfinite(chip_kdp) & (np.abs(chip_kdp) > 0.05)
    az = np.linspace(float(chip["az_lower"][0]), float(chip["az_upper"][0]), chip["DBZ"].shape[1]).astype(np.float32)
    rg = np.linspace(float(chip["rng_lower"][0]), float(chip["rng_upper"][0]), chip["DBZ"].shape[2]).astype(np.float32)

    dt = xd.io.open_nexradlevel2_datatree(str(raw_path))
    if args.auto_align:
        sweep_candidates = get_lowest_kdp_sweeps(dt, max_count=2)
        pyart_fields = {sw: build_kdp_from_pyart(raw_path, sw, az, rg) for sw in sweep_candidates}
        if all(pyart_fields[sw] is not None for sw in sweep_candidates):
            best = choose_best_kdp_alignment(
                dt,
                chip_kdp,
                az,
                rg,
                sweeps=sweep_candidates,
                raw_kdp_by_sweep=pyart_fields,
            )
            source_name = "pyart"
        else:
            best = choose_best_kdp_alignment(dt, chip_kdp, az, rg, sweeps=sweep_candidates)
            source_name = "phidp-grad"
        raw_kdp = best["raw_kdp"]
        meta = (
            f"{source_name}, auto sw={best['sweep']}, sweeps={sweep_candidates}, sign={int(best['sign'])}, az_flip={best['az_flip']}, "
            f"r_flip={best['r_flip']}, roll={best['az_roll']}, smooth={best['smooth_window']}"
        )
    else:
        ds = dt["sweep_0"].to_dataset().sortby("azimuth")
        raw_kdp = build_kdp_from_pyart(raw_path, 0, az, rg)
        if raw_kdp is not None:
            if args.kdp_az_flip:
                raw_kdp = raw_kdp[::-1, :]
            if args.kdp_r_flip:
                raw_kdp = raw_kdp[:, ::-1]
            if args.kdp_az_roll:
                raw_kdp = np.roll(raw_kdp, args.kdp_az_roll, axis=0)
            raw_kdp = args.kdp_sign * raw_kdp
            source_name = "pyart"
        else:
            raw_kdp = build_kdp_from_phidp(
                ds,
                az,
                rg,
                sign=args.kdp_sign,
                az_flip=args.kdp_az_flip,
                r_flip=args.kdp_r_flip,
                az_roll=args.kdp_az_roll,
                smooth_window=args.kdp_smooth_window,
            )
            source_name = "phidp-grad"
        meta = (
            f"{source_name}, manual sign={int(args.kdp_sign)}, az_flip={args.kdp_az_flip}, "
            f"r_flip={args.kdp_r_flip}, roll={args.kdp_az_roll}, smooth={args.kdp_smooth_window}"
        )

    raw_kdp = raw_kdp.copy()
    raw_kdp[~chip_valid] = np.nan

    azr = np.deg2rad(az)
    rr = rg / 1000.0
    tt, rr2 = np.meshgrid(azr, rr, indexing="ij")
    r0 = float(chip["rng_lower"][0]) / 1000.0
    cmap, norm = get_cmap("KDP")

    fig = plt.figure(figsize=(6, 8), facecolor="0.92")
    ax1 = fig.add_subplot(2, 1, 1, polar=True)
    ax1.pcolormesh(tt, rr2 - r0, chip_kdp, shading="nearest", cmap=cmap, norm=norm)
    ax1.set_theta_zero_location("N")
    ax1.set_theta_direction(-1)
    ax1.set_rorigin(-r0)
    ax1.grid(False)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_title("KDP chip")

    ax2 = fig.add_subplot(2, 1, 2, polar=True)
    ax2.pcolormesh(tt, rr2 - r0, raw_kdp, shading="nearest", cmap=cmap, norm=norm)
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    ax2.set_rorigin(-r0)
    ax2.grid(False)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_title(f"KDP raw chip-mask clipped ({meta})")

    fig.text(0.5, 0.01, f"{chip_path.name} vs {raw_path.name}", ha="center")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"SAVED {args.output}")


if __name__ == "__main__":
    main()
