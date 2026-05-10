import numpy as np
import pytest
import xarray as xr

from tornet.data.kdp import (
    KDP_SOURCE_PHIDP_GRADIENT,
    build_target_axis,
    build_kdp_from_phidp_sweep,
    build_kdp_tilt_stack,
    build_tornet_tilt_sweep_pairs,
    compute_kdp_comparison_metrics,
    get_kdp_match_failures,
    mask_kdp_for_comparison,
    require_dataset_fields,
    require_datatree_sweeps,
    sanitize_velocity_sweep_dataset,
)


class _Node:
    def __init__(self, ds: xr.Dataset):
        self._ds = ds

    def to_dataset(self) -> xr.Dataset:
        return self._ds


def _make_sweep(phidp_slope_deg_per_km: float) -> xr.Dataset:
    az = np.array([0.0, 90.0, 180.0], dtype=np.float32)
    rg = np.array([1000.0, 2000.0, 3000.0, 4000.0], dtype=np.float32)
    phidp = (phidp_slope_deg_per_km * (rg / 1000.0))[None, :] * np.ones((len(az), 1), dtype=np.float32)
    dbz = np.full((len(az), len(rg)), 35.0, dtype=np.float32)
    rhohv = np.full((len(az), len(rg)), 0.9, dtype=np.float32)
    return xr.Dataset(
        {
            "PHIDP": (("azimuth", "range"), phidp.astype(np.float32)),
            "DBZH": (("azimuth", "range"), dbz),
            "RHOHV": (("azimuth", "range"), rhohv),
        },
        coords={"azimuth": az, "range": rg},
    )


def test_build_tornet_tilt_sweep_pairs():
    assert build_tornet_tilt_sweep_pairs(2) == [(0, 1), (2, 3)]


def test_build_target_axis_center_sampling():
    centered = build_target_axis(-13.0, 47.0, 120, center_sampling=True)
    assert centered.shape == (120,)
    assert centered[0] == pytest.approx(-12.75)
    assert centered[-1] == pytest.approx(46.75)

    endpoints = build_target_axis(-13.0, 47.0, 120, center_sampling=False)
    assert endpoints[0] == pytest.approx(-13.0)
    assert endpoints[-1] == pytest.approx(47.0)


def test_require_datatree_sweeps_fails_loudly():
    dt = {"sweep_0": _Node(_make_sweep(1.0))}
    with pytest.raises(ValueError, match="missing required sweeps"):
        require_datatree_sweeps(dt, [0, 2], context="unit-test")


def test_require_dataset_fields_fails_loudly():
    with pytest.raises(ValueError, match="missing required fields"):
        require_dataset_fields(_make_sweep(1.0), ["PHIDP", "MISSING"], context="unit-test")


def test_sanitize_velocity_sweep_dataset_masks_invalid_width():
    az = np.array([0.0, 1.0], dtype=np.float32)
    rg = np.array([1000.0, 2000.0], dtype=np.float32)
    vel = np.array([[-1.0, -64.5], [2.0, 3.0]], dtype=np.float32)
    width = np.array([[0.5, -64.5], [1.0, -1.0]], dtype=np.float32)
    ds = xr.Dataset(
        {
            "VRADH": (("azimuth", "range"), vel, {"_FillValue": -64.5}),
            "WRADH": (("azimuth", "range"), width, {"_FillValue": -64.5}),
        },
        coords={"azimuth": az, "range": rg},
    )
    out = sanitize_velocity_sweep_dataset(ds)
    out_vel = out["VRADH"].values
    out_width = out["WRADH"].values
    assert np.isnan(out_vel[0, 1]) and np.isnan(out_width[0, 1])
    assert np.isnan(out_vel[1, 1]) and np.isnan(out_width[1, 1])
    assert out_width[0, 0] == pytest.approx(0.5)


def test_build_kdp_from_phidp_sweep_gradient():
    ds = _make_sweep(2.0)
    az = np.array([0.0, 90.0, 180.0], dtype=np.float32)
    rg = np.array([1000.0, 2000.0, 3000.0, 4000.0], dtype=np.float32)
    out = build_kdp_from_phidp_sweep(ds, az, rg, interp_method="nearest", smooth_window=1)
    assert out.shape == (3, 4)
    assert np.allclose(out, 2.0, atol=1e-5)


def test_build_kdp_tilt_stack_phidp_mapping_and_stats():
    dt = {
        "sweep_0": _Node(_make_sweep(1.0)),
        "sweep_1": _Node(_make_sweep(10.0)),
        "sweep_2": _Node(_make_sweep(3.0)),
        "sweep_3": _Node(_make_sweep(10.0)),
    }
    az = np.array([0.0, 90.0, 180.0], dtype=np.float32)
    rg = np.array([1000.0, 2000.0, 3000.0, 4000.0], dtype=np.float32)

    kdp = build_kdp_tilt_stack(
        dt,
        az_target=az,
        range_target=rg,
        n_tilts=2,
        kdp_source=KDP_SOURCE_PHIDP_GRADIENT,
        phidp_interp_method="nearest",
        phidp_smooth_window=1,
        context="unit-test",
    )
    assert kdp.shape == (3, 4, 2)
    assert np.allclose(kdp[:, :, 0], 1.0, atol=1e-5)
    assert np.allclose(kdp[:, :, 1], 3.0, atol=1e-5)


def test_shared_mask_and_metrics_and_thresholds():
    chip = np.array([[0.1, 0.2], [np.nan, 0.01]], dtype=np.float32)
    raw = np.array([[0.1, 0.3], [0.5, 0.01]], dtype=np.float32)
    dbz = np.full((2, 2), 30.0, dtype=np.float32)
    rhohv = np.full((2, 2), 0.9, dtype=np.float32)

    chip_cmp, raw_cmp, mask = mask_kdp_for_comparison(chip, raw, dbz=dbz, rhohv=rhohv, min_abs_kdp=0.05)
    assert mask.sum() == 2
    assert np.isnan(chip_cmp[1, 1]) and np.isnan(raw_cmp[1, 1])

    metrics = compute_kdp_comparison_metrics(chip_cmp, raw_cmp, active_threshold=0.05)
    assert metrics["iou"] == pytest.approx(1.0)
    assert metrics["mad"] == pytest.approx(0.05)
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.005))
    assert metrics["centroid_offset_px"] == pytest.approx(0.0)

    assert get_kdp_match_failures(
        metrics,
        min_iou=0.9,
        max_mad=0.1,
        max_rmse=0.1,
        max_centroid_offset_px=0.5,
    ) == []
