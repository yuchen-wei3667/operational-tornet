"""
Utilities to build TorNet-style tensors from NEXRAD Level-II files using xradar.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import xarray as xr

from tornet.data.kdp import (
    KDP_SOURCE_PYART_VULPIANI,
    build_target_axis,
    build_kdp_tilt_stack,
    build_tornet_tilt_sweep_pairs,
    extract_pyart_sweep_field,
    interp_with_azimuth_wrap,
    require_dataset_fields,
    require_datatree_sweeps,
    require_radar_sweeps,
    sanitize_velocity_sweep_dataset,
)


def read_nexrad_v06_to_tornet(
    scan_path: str,
    n_sweeps: int = 2,
    az_lower_deg: float = -13.0,
    az_upper_deg: float = 47.0,
    rng_lower_m: float = 51488.0,
    rng_upper_m: float = 111488.0,
    n_azimuth: int = 120,
    n_range: int = 240,
    interp_method: str = "linear",
    kdp_interp_method: str = "nearest",
    use_dealiased_velocity: bool = True,
    dealias_centered: bool = True,
    kdp_windsize: int = 9,
    kdp_prefilter: bool = True,
    kdp_source: str = KDP_SOURCE_PYART_VULPIANI,
) -> Dict[str, np.ndarray]:
    """Reads a Level-II V06 file into a TorNet-compatible dictionary.

    This recreates TorNet-like chip geometry and channel layout:
    - fixed azimuth/range limits and resolution (default: 120x240)
    - tilt channels built from paired sweeps (0/1), (2/3), ...
    - reflectivity/polarimetric fields from even sweeps
    - velocity/width fields from odd sweeps
    - KDP from a shared builder (`pyart_vulpiani` by default)

    Output tensor layout matches `read_file` convention: [time, azimuth, range, tilt].
    """
    import xradar as xd

    dt = xd.io.open_nexradlevel2_datatree(scan_path)

    sweep_pairs = build_tornet_tilt_sweep_pairs(n_sweeps)
    required_sweeps = sorted({idx for refl, vel in sweep_pairs for idx in (refl, vel)})
    require_datatree_sweeps(dt, required_sweeps, context="read_nexrad_v06_to_tornet")

    # Retrieve dealiased velocity with Py-ART.
    import pyart

    radar = pyart.io.read_nexrad_archive(scan_path)
    require_radar_sweeps(radar, required_sweeps, context="read_nexrad_v06_to_tornet")
    if use_dealiased_velocity:
        dealiased = pyart.correct.dealias_region_based(
            radar,
            vel_field="velocity",
            centered=dealias_centered,
            keep_original=False,
        )
        vel_data = np.ma.filled(dealiased["data"], np.nan).astype(np.float32)
    else:
        vel_data = np.ma.filled(radar.fields["velocity"]["data"], np.nan).astype(np.float32)

    dbz = []
    vel = []
    kdp = []
    rhohv = []
    zdr = []
    width = []

    az_lowers = []
    az_uppers = []
    rng_lowers = []
    rng_uppers = []

    az_target = build_target_axis(az_lower_deg, az_upper_deg, n_azimuth, center_sampling=True)
    range_target = build_target_axis(rng_lower_m, rng_upper_m, n_range, center_sampling=True)

    kdp_stack = build_kdp_tilt_stack(
        dt,
        az_target=az_target,
        range_target=range_target,
        n_tilts=n_sweeps,
        kdp_source=kdp_source,
        radar=radar,
        kdp_interp_method=kdp_interp_method,
        pyart_kdp_windsize=kdp_windsize,
        pyart_kdp_prefilter=kdp_prefilter,
        context="read_nexrad_v06_to_tornet",
    )

    for tilt_idx, (refl_sweep, vel_sweep) in enumerate(sweep_pairs):
        refl_ds = dt[f"sweep_{refl_sweep}"].to_dataset()
        vel_ds = dt[f"sweep_{vel_sweep}"].to_dataset()
        require_dataset_fields(
            refl_ds,
            ["DBZH", "ZDR", "RHOHV"],
            context=f"read_nexrad_v06_to_tornet:sweep_{refl_sweep}",
        )
        require_dataset_fields(
            vel_ds,
            ["WRADH"],
            context=f"read_nexrad_v06_to_tornet:sweep_{vel_sweep}",
        )
        vel_ds = sanitize_velocity_sweep_dataset(vel_ds)

        refl_interp = interp_with_azimuth_wrap(refl_ds, az_target, range_target, interp_method)
        vel_interp = interp_with_azimuth_wrap(vel_ds, az_target, range_target, interp_method)

        dbz.append(refl_interp["DBZH"].values.astype(np.float32))
        zdr.append(refl_interp["ZDR"].values.astype(np.float32))
        rhohv.append(refl_interp["RHOHV"].values.astype(np.float32))

        kdp.append(kdp_stack[:, :, tilt_idx].astype(np.float32))

        vaz, vrg, vfld = extract_pyart_sweep_field(radar, vel_data, vel_sweep)
        vds = xr.Dataset({"vel": (("azimuth", "range"), vfld)}, coords={"azimuth": vaz, "range": vrg})
        v_interp = interp_with_azimuth_wrap(vds, az_target, range_target, interp_method)
        vel.append(v_interp["vel"].values.astype(np.float32))
        width.append(vel_interp["WRADH"].values.astype(np.float32))

    az_lowers.append(float(np.nanmin(az_target)))
    az_uppers.append(float(np.nanmax(az_target)))
    rng_lowers.append(float(np.nanmin(range_target)))
    rng_uppers.append(float(np.nanmax(range_target)))

    # Stack as [time=1, azimuth, range, sweep]
    out: Dict[str, np.ndarray] = {
        "DBZ": np.stack(dbz, axis=-1)[None, ...],
        "VEL": np.stack(vel, axis=-1)[None, ...],
        "KDP": np.stack(kdp, axis=-1)[None, ...],
        "RHOHV": np.stack(rhohv, axis=-1)[None, ...],
        "ZDR": np.stack(zdr, axis=-1)[None, ...],
        "WIDTH": np.stack(width, axis=-1)[None, ...],
        "range_folded_mask": np.zeros_like(np.stack(vel, axis=-1)[None, ...], dtype=np.float32),
        "label": np.array([[0]], dtype=np.int64),
        "category": np.array([[1]], dtype=np.int64),
        "event_id": np.array([[0]], dtype=np.int64),
        "ef_number": np.array([[-1]], dtype=np.int64),
        "az_lower": np.array([min(az_lowers)], dtype=np.float32),
        "az_upper": np.array([max(az_uppers)], dtype=np.float32),
        "rng_lower": np.array([min(rng_lowers)], dtype=np.float32),
        "rng_upper": np.array([max(rng_uppers)], dtype=np.float32),
        "time": np.array([[0]], dtype=np.int64),
        "tornado_start_time": np.array([[0]], dtype=np.int64),
        "tornado_end_time": np.array([[0]], dtype=np.int64),
    }

    return out
