"""
Utilities to build TorNet-style tensors from NEXRAD Level-II files using xradar.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import xarray as xr


def _sort_azimuth(ds: xr.Dataset) -> xr.Dataset:
    az = ds["azimuth"].values
    order = np.argsort(az)
    return ds.isel(azimuth=order)


def _interp_with_azimuth_wrap(
    ds: xr.Dataset,
    az_target: np.ndarray,
    range_target: np.ndarray,
    interp_method: str,
) -> xr.Dataset:
    """Interpolate polar data with explicit 0/360 azimuth wrap handling."""
    ds0 = _sort_azimuth(ds)

    # Extend azimuth coordinate by +/-360 so windows crossing 0 deg interpolate
    # continuously (e.g., 317..377 deg).
    ds_m = ds0.assign_coords(azimuth=(ds0["azimuth"] - 360.0))
    ds_p = ds0.assign_coords(azimuth=(ds0["azimuth"] + 360.0))
    ds_ext = xr.concat([ds_m, ds0, ds_p], dim="azimuth").sortby("azimuth")

    return ds_ext.interp(
        azimuth=xr.DataArray(az_target.astype(np.float32), dims=["azimuth"]),
        range=xr.DataArray(range_target.astype(np.float32), dims=["range"]),
        method=interp_method,
    )


def _kdp_from_phidp(phidp_deg: np.ndarray, range_m: np.ndarray) -> np.ndarray:
    range_km = np.maximum(range_m / 1000.0, 1e-3)
    return np.gradient(phidp_deg, range_km, axis=1)


def _pyart_sweep_field(radar, field_data: np.ndarray, sweep_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sorted (azimuth_deg, range_m, field_2d) for one sweep from a Py-ART Radar."""
    s0 = int(radar.sweep_start_ray_index["data"][sweep_idx])
    s1 = int(radar.sweep_end_ray_index["data"][sweep_idx]) + 1
    az = np.asarray(radar.azimuth["data"][s0:s1], dtype=np.float32)
    rg = np.asarray(radar.range["data"], dtype=np.float32)
    fld_src = field_data[s0:s1, :]
    if np.ma.isMaskedArray(fld_src):
        fld = np.ma.filled(fld_src, np.nan).astype(np.float32)
    else:
        fld = np.asarray(fld_src, dtype=np.float32)

    order = np.argsort(az)
    return az[order], rg, fld[order]


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
) -> Dict[str, np.ndarray]:
    """Reads a Level-II V06 file into a TorNet-compatible dictionary.

    This recreates TorNet-like chip geometry and channel layout:
    - fixed azimuth/range limits and resolution (default: 120x240)
    - two tilt channels built from paired sweeps (0/1) and (2/3)
    - reflectivity/polarimetric fields from even sweeps
    - velocity/width fields from odd sweeps

    Output tensor layout matches `read_file` convention: [time, azimuth, range, tilt].
    """
    import xradar as xd

    dt = xd.io.open_nexradlevel2_datatree(scan_path)

    # Retrieve dealiased velocity and KDP with Py-ART
    import pyart

    radar = pyart.io.read_nexrad_archive(scan_path)
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

    kdp_dict, _ = pyart.retrieve.kdp_vulpiani(
        radar,
        phidp_field="differential_phase",
        band="S",
        windsize=kdp_windsize,
        prefilter_psidp=kdp_prefilter,
    )
    kdp_data = np.ma.filled(kdp_dict["data"], np.nan).astype(np.float32)

    if n_sweeps != 2:
        raise ValueError("This loader currently supports n_sweeps=2 only.")

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

    # Match TorNet chip center-sampling convention (half-bin inset from limits)
    if n_azimuth > 1:
        az_step = (az_upper_deg - az_lower_deg) / float(n_azimuth)
    else:
        az_step = 0.0
    if n_range > 1:
        rg_step = (rng_upper_m - rng_lower_m) / float(n_range)
    else:
        rg_step = 0.0

    az_target = np.linspace(
        az_lower_deg + 0.5 * az_step,
        az_upper_deg - 0.5 * az_step,
        n_azimuth,
        dtype=np.float32,
    )
    range_target = np.linspace(
        rng_lower_m + 0.5 * rg_step,
        rng_upper_m - 0.5 * rg_step,
        n_range,
        dtype=np.float32,
    )

    for k in range(n_sweeps):
        refl_ds = dt[f"sweep_{2*k}"].to_dataset()
        vel_ds = dt[f"sweep_{2*k+1}"].to_dataset()

        refl_interp = _interp_with_azimuth_wrap(refl_ds, az_target, range_target, interp_method)
        vel_interp = _interp_with_azimuth_wrap(vel_ds, az_target, range_target, interp_method)

        dbz.append(refl_interp["DBZH"].values.astype(np.float32))
        zdr.append(refl_interp["ZDR"].values.astype(np.float32))
        rhohv.append(refl_interp["RHOHV"].values.astype(np.float32))

        # KDP from Py-ART retrieval on the corresponding reflectivity sweep
        kaz, krg, kfld = _pyart_sweep_field(radar, kdp_data, 2 * k)
        kds = xr.Dataset({"kdp": (("azimuth", "range"), kfld)}, coords={"azimuth": kaz, "range": krg})
        k_interp = _interp_with_azimuth_wrap(kds, az_target, range_target, kdp_interp_method)
        kdp.append(k_interp["kdp"].values.astype(np.float32))

        vaz, vrg, vfld = _pyart_sweep_field(radar, vel_data, 2 * k + 1)
        vds = xr.Dataset({"vel": (("azimuth", "range"), vfld)}, coords={"azimuth": vaz, "range": vrg})
        v_interp = _interp_with_azimuth_wrap(vds, az_target, range_target, interp_method)
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
