"""
Shared KDP processing utilities for TorNet-style raw/chip comparisons.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

KDP_SOURCE_PYART_VULPIANI = "pyart_vulpiani"
KDP_SOURCE_PHIDP_GRADIENT = "phidp_gradient"
VALID_KDP_SOURCES = (KDP_SOURCE_PYART_VULPIANI, KDP_SOURCE_PHIDP_GRADIENT)
_FILL_KEYS = ("_FillValue", "missing_value", "fill_value")


def build_tornet_tilt_sweep_pairs(n_tilts: int) -> list[tuple[int, int]]:
    """Return explicit (reflectivity_sweep, velocity_sweep) pairs per tilt."""
    if n_tilts < 1:
        raise ValueError(f"n_tilts must be >= 1, got {n_tilts}.")
    return [(2 * tilt, 2 * tilt + 1) for tilt in range(n_tilts)]


def reflectivity_sweep_indices(n_tilts: int) -> list[int]:
    return [pair[0] for pair in build_tornet_tilt_sweep_pairs(n_tilts)]


def _sweep_key_set(keys: Iterable[str]) -> set[str]:
    return {str(key) for key in keys}


def build_target_axis(
    lower: float,
    upper: float,
    n_bins: int,
    *,
    center_sampling: bool = True,
) -> np.ndarray:
    """Build a target axis using endpoint or center-bin sampling."""
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}.")

    if center_sampling and n_bins > 1:
        step = (upper - lower) / float(n_bins)
        start = lower + 0.5 * step
        stop = upper - 0.5 * step
    else:
        start = lower
        stop = upper

    return np.linspace(start, stop, n_bins, dtype=np.float32)


def require_datatree_sweeps(
    datatree: Mapping[str, Any],
    sweep_indices: Sequence[int],
    *,
    context: str = "",
) -> None:
    """Validate all requested sweep_N keys are present in a DataTree-like mapping."""
    keys = _sweep_key_set(datatree.keys())
    missing = [idx for idx in sweep_indices if f"sweep_{idx}" not in keys]
    if missing:
        prefix = f"{context}: " if context else ""
        raise ValueError(f"{prefix}missing required sweeps in datatree: {missing}")


def require_radar_sweeps(
    radar: Any,
    sweep_indices: Sequence[int],
    *,
    context: str = "",
) -> None:
    """Validate requested sweep indices exist in a Py-ART Radar object."""
    nsweeps = int(getattr(radar, "nsweeps", 0))
    missing = [idx for idx in sweep_indices if idx < 0 or idx >= nsweeps]
    if missing:
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}missing required sweeps in radar volume: {missing} (nsweeps={nsweeps})"
        )


def require_dataset_fields(ds: xr.Dataset, fields: Sequence[str], *, context: str = "") -> None:
    missing = [field for field in fields if field not in ds]
    if missing:
        prefix = f"{context}: " if context else ""
        raise ValueError(f"{prefix}missing required fields in dataset: {missing}")


def sort_azimuth(ds: xr.Dataset) -> xr.Dataset:
    az = ds["azimuth"].values
    order = np.argsort(az)
    return ds.isel(azimuth=order)


def _iter_fill_values(da: xr.DataArray) -> list[float]:
    values: list[float] = []
    for key in _FILL_KEYS:
        for mapping in (da.attrs, da.encoding):
            raw = mapping.get(key)
            if raw is None:
                continue
            arr = np.atleast_1d(raw)
            for item in arr:
                try:
                    v = float(item)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(v):
                    values.append(v)
    return values


def mask_fill_values(da: xr.DataArray) -> xr.DataArray:
    out = da.astype(np.float32)
    for fill_value in _iter_fill_values(da):
        out = out.where(out != np.float32(fill_value), np.nan)
    return out


def sanitize_velocity_sweep_dataset(
    ds: xr.Dataset,
    *,
    velocity_field: str = "VRADH",
    width_field: str = "WRADH",
) -> xr.Dataset:
    """Convert sweep velocity masked/fill values into NaN and preserve through interpolation."""
    require_dataset_fields(ds, [width_field], context="sanitize_velocity_sweep_dataset")

    out = ds.copy(deep=False)
    width = mask_fill_values(out[width_field]).astype(np.float32)
    invalid = ~np.isfinite(width) | (width < 0.0)
    out[width_field] = width.where(~invalid, np.nan).astype(np.float32)

    if velocity_field in out:
        velocity = mask_fill_values(out[velocity_field]).astype(np.float32)
        out[velocity_field] = velocity.where(~invalid, np.nan).astype(np.float32)

    return out


def interp_with_azimuth_wrap(
    ds: xr.Dataset,
    az_target: np.ndarray,
    range_target: np.ndarray,
    interp_method: str,
) -> xr.Dataset:
    """Interpolate polar data with explicit 0/360 azimuth wrap handling."""
    ds0 = sort_azimuth(ds)

    ds_m = ds0.assign_coords(azimuth=(ds0["azimuth"] - 360.0))
    ds_p = ds0.assign_coords(azimuth=(ds0["azimuth"] + 360.0))
    ds_ext = xr.concat([ds_m, ds0, ds_p], dim="azimuth", data_vars="all").sortby("azimuth")

    return ds_ext.interp(
        azimuth=xr.DataArray(az_target.astype(np.float32), dims=["azimuth"]),
        range=xr.DataArray(range_target.astype(np.float32), dims=["range"]),
        method=interp_method,
    )


def extract_pyart_sweep_field(
    radar: Any,
    field_data: np.ndarray,
    sweep_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def retrieve_pyart_kdp_field(
    radar: Any,
    *,
    phidp_field: str = "differential_phase",
    band: str = "S",
    windsize: int = 9,
    prefilter_psidp: bool = True,
) -> np.ndarray:
    import pyart

    kdp_dict, _ = pyart.retrieve.kdp_vulpiani(
        radar,
        phidp_field=phidp_field,
        band=band,
        windsize=windsize,
        prefilter_psidp=prefilter_psidp,
    )
    return np.ma.filled(kdp_dict["data"], np.nan).astype(np.float32)


def build_kdp_from_pyart_field(
    radar: Any,
    kdp_field_data: np.ndarray,
    sweep_idx: int,
    az_target: np.ndarray,
    range_target: np.ndarray,
    *,
    interp_method: str = "nearest",
) -> np.ndarray:
    kaz, krg, kfld = extract_pyart_sweep_field(radar, kdp_field_data, sweep_idx)
    kds = xr.Dataset({"kdp": (("azimuth", "range"), kfld)}, coords={"azimuth": kaz, "range": krg})
    k_interp = interp_with_azimuth_wrap(kds, az_target, range_target, interp_method)
    return k_interp["kdp"].values.astype(np.float32)


def build_kdp_from_phidp_sweep(
    ds_sweep: xr.Dataset,
    az_target: np.ndarray,
    range_target: np.ndarray,
    *,
    interp_method: str = "linear",
    smooth_window: int = 9,
) -> np.ndarray:
    if "PHIDP" not in ds_sweep:
        raise ValueError("PHIDP field is required for phidp_gradient KDP source.")

    ph = interp_with_azimuth_wrap(ds_sweep, az_target, range_target, interp_method)["PHIDP"].values.astype(
        np.float32
    )
    if smooth_window > 1:
        kernel = np.ones(int(smooth_window), dtype=np.float32) / float(smooth_window)
        ph = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, ph)

    range_km = np.maximum(range_target.astype(np.float32) / 1000.0, 1e-3)
    return np.gradient(ph, range_km, axis=1).astype(np.float32)


def build_kdp_tilt_stack(
    datatree: Mapping[str, Any],
    az_target: np.ndarray,
    range_target: np.ndarray,
    *,
    n_tilts: int,
    kdp_source: str = KDP_SOURCE_PYART_VULPIANI,
    radar: Any = None,
    kdp_interp_method: str = "nearest",
    phidp_interp_method: str = "linear",
    phidp_smooth_window: int = 9,
    pyart_kdp_windsize: int = 9,
    pyart_kdp_prefilter: bool = True,
    context: str = "build_kdp_tilt_stack",
) -> np.ndarray:
    if kdp_source not in VALID_KDP_SOURCES:
        raise ValueError(f"{context}: invalid kdp_source={kdp_source!r}; expected one of {VALID_KDP_SOURCES}.")

    refl_sweeps = reflectivity_sweep_indices(n_tilts)
    require_datatree_sweeps(datatree, refl_sweeps, context=context)

    kdp_by_tilt: list[np.ndarray] = []
    if kdp_source == KDP_SOURCE_PYART_VULPIANI:
        if radar is None:
            raise ValueError(f"{context}: radar is required for kdp_source={KDP_SOURCE_PYART_VULPIANI!r}.")
        require_radar_sweeps(radar, refl_sweeps, context=context)
        kdp_field = retrieve_pyart_kdp_field(
            radar,
            windsize=pyart_kdp_windsize,
            prefilter_psidp=pyart_kdp_prefilter,
        )
        for sweep_idx in refl_sweeps:
            kdp_by_tilt.append(
                build_kdp_from_pyart_field(
                    radar,
                    kdp_field,
                    sweep_idx,
                    az_target,
                    range_target,
                    interp_method=kdp_interp_method,
                )
            )
    else:
        for sweep_idx in refl_sweeps:
            ds = datatree[f"sweep_{sweep_idx}"].to_dataset()
            kdp_by_tilt.append(
                build_kdp_from_phidp_sweep(
                    ds,
                    az_target,
                    range_target,
                    interp_method=phidp_interp_method,
                    smooth_window=phidp_smooth_window,
                )
            )

    return np.stack(kdp_by_tilt, axis=-1).astype(np.float32)


def mask_kdp_for_comparison(
    chip_kdp: np.ndarray,
    raw_kdp: np.ndarray,
    *,
    dbz: np.ndarray | None = None,
    rhohv: np.ndarray | None = None,
    min_abs_kdp: float = 0.05,
    min_dbz: float = 20.0,
    min_rhohv: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a shared finite/QC policy to chip and raw KDP before comparison."""
    mask = np.isfinite(chip_kdp) & np.isfinite(raw_kdp)

    if dbz is not None:
        mask &= np.isfinite(dbz) & (dbz >= min_dbz)
    if rhohv is not None:
        mask &= np.isfinite(rhohv) & (rhohv >= min_rhohv)

    mask &= np.abs(chip_kdp) >= min_abs_kdp
    mask &= np.abs(raw_kdp) >= min_abs_kdp

    chip_masked = np.where(mask, chip_kdp, np.nan).astype(np.float32)
    raw_masked = np.where(mask, raw_kdp, np.nan).astype(np.float32)
    return chip_masked, raw_masked, mask


def compute_kdp_comparison_metrics(
    chip_kdp: np.ndarray,
    raw_kdp: np.ndarray,
    *,
    active_threshold: float = 0.05,
) -> dict[str, float]:
    overlap = np.isfinite(chip_kdp) & np.isfinite(raw_kdp)
    overlap_count = int(overlap.sum())

    if overlap_count == 0:
        mad = np.nan
        rmse = np.nan
    else:
        diff = chip_kdp[overlap] - raw_kdp[overlap]
        mad = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(np.square(diff))))

    chip_active = np.isfinite(chip_kdp) & (np.abs(chip_kdp) >= active_threshold)
    raw_active = np.isfinite(raw_kdp) & (np.abs(raw_kdp) >= active_threshold)
    inter = int((chip_active & raw_active).sum())
    union = int((chip_active | raw_active).sum())
    iou = float(inter / union) if union else np.nan

    chip_pts = np.argwhere(chip_active)
    raw_pts = np.argwhere(raw_active)
    if len(chip_pts) and len(raw_pts):
        centroid_offset_px = float(np.linalg.norm(chip_pts.mean(axis=0) - raw_pts.mean(axis=0)))
    else:
        centroid_offset_px = np.inf

    return {
        "iou": iou,
        "mad": mad,
        "rmse": rmse,
        "centroid_offset_px": centroid_offset_px,
        "overlap_count": overlap_count,
        "intersection_count": inter,
        "union_count": union,
    }


def get_kdp_match_failures(
    metrics: Mapping[str, float],
    *,
    min_iou: float,
    max_mad: float,
    max_rmse: float,
    max_centroid_offset_px: float,
) -> list[str]:
    failures: list[str] = []

    iou = float(metrics["iou"])
    mad = float(metrics["mad"])
    rmse = float(metrics["rmse"])
    centroid_offset = float(metrics["centroid_offset_px"])

    if not np.isfinite(iou) or iou < min_iou:
        failures.append(f"iou={iou:.4f} < min_iou={min_iou:.4f}")
    if not np.isfinite(mad) or mad > max_mad:
        failures.append(f"mad={mad:.4f} > max_mad={max_mad:.4f}")
    if not np.isfinite(rmse) or rmse > max_rmse:
        failures.append(f"rmse={rmse:.4f} > max_rmse={max_rmse:.4f}")
    if not np.isfinite(centroid_offset) or centroid_offset > max_centroid_offset_px:
        failures.append(
            f"centroid_offset_px={centroid_offset:.4f} > max_centroid_offset_px={max_centroid_offset_px:.4f}"
        )

    return failures
