"""
Offline unit tests for tornet.nexrad (Level-II/III -> TorNet inputs).

The expected values encode conventions verified against released TorNet samples
(see tornet/nexrad/README.md).  End-to-end checks against real TorNet files are done
by scripts/nexrad/validate_against_tornet.py (needs network + TorNet data).
"""
import datetime

import numpy as np
import pytest

from tornet.data.constants import ALL_VARIABLES
from tornet.nexrad import level2 as l2
from tornet.nexrad.kdp import kdp_from_level2
from tornet.nexrad.level3 import level3_to_tornet_grid
from tornet.nexrad.pipeline import (CHIP_N_AZ, CHIP_N_RNG, TorNetVolume, chip_indices,
                                    chip_limits, extract_chip, netcdf_quantize, stack_frames,
                                    tile_centers)


# --------------------------------------------------------------------- fake Level-II
class _FakeL2File:
    """Minimal stand-in for pyart's NEXRADLevel2File (raw-code access only)."""

    def __init__(self, sweeps):
        # sweeps: list of dict(az=[n], moments={name: (codes[n, g], scale, offset, first_gate, word)})
        self._sweeps = sweeps
        self.scan_msgs, self.radial_records = [], []
        k = 0
        for s in sweeps:
            n = len(s["az"])
            self.scan_msgs.append(list(range(k, k + n)))
            for _ in range(n):
                rec = {m: {"ngates": c.shape[1], "first_gate": fg, "gate_spacing": 250,
                           "scale": sc, "offset": off, "word_size": w}
                       for m, (c, sc, off, fg, w) in s["moments"].items()}
                self.radial_records.append(rec)
            k += n

    def get_data(self, moment, max_ngates, scans, raw_data=True):
        return self._sweeps[scans[0]]["moments"][moment][0]

    def get_azimuth_angles(self, scans):
        return np.asarray(self._sweeps[scans[0]]["az"], dtype=np.float64)


def _volume(sweep_defs, t0=datetime.datetime(2022, 5, 1, 0, 0, 0)):
    f = _FakeL2File(sweep_defs)
    infos = []
    for i, s in enumerate(sweep_defs):
        infos.append(l2.SweepInfo(index=i, elevation=s["elev"], moments=tuple(s["moments"]),
                                  time=t0 + datetime.timedelta(seconds=20 * i), nyquist=s.get("nyq", 25.0),
                                  n_radials=len(s["az"]), complete=s.get("complete", True)))
    return l2.Level2Volume(site="KXXX", latitude=35.0, longitude=-97.0, vcp=212, time=t0,
                           sweeps=infos, _file=f)


def _super_res_az(start=0.25):
    return (start + 0.5 * np.arange(720)) % 360


def _split_cut(elev, n_gates=40, zdr_word=8, nyq=25.0):
    rng = np.random.default_rng(int(elev * 10))
    ref = rng.integers(2, 200, size=(720, n_gates))
    ref[:, :3] = 0                                        # below threshold
    zdr = rng.integers(2, 255, size=(720, n_gates))
    zdr[:, :3] = 0
    rho = rng.integers(2, 255, size=(720, n_gates))
    phi = rng.integers(2, 1000, size=(720, n_gates))
    vel = rng.integers(2, 255, size=(720, n_gates))
    vel[0, 10] = 1                                        # range folded in VEL
    sw = rng.integers(2, 255, size=(720, n_gates))
    sw[1, 11] = 1                                         # range folded flagged only in SW
    zdr_scale, zdr_off = (16.0, 128.0) if zdr_word == 8 else (32.0, 418.0)
    cs = {"elev": elev - 0.02, "az": _super_res_az(),
          "moments": {"REF": (ref, 2.0, 66.0, 2125, 8), "ZDR": (zdr, zdr_scale, zdr_off, 2125, zdr_word),
                      "RHO": (rho, 300.0, -60.5, 2125, 8), "PHI": (phi, 2.8361, 2.0, 2125, 16)}}
    cd = {"elev": elev - 0.02, "az": _super_res_az(0.2), "nyq": nyq,
          "moments": {"REF": (ref, 2.0, 66.0, 2125, 8), "VEL": (vel, 2.0, 129.0, 2125, 8),
                      "SW": (sw, 2.0, 129.0, 2125, 8)}}
    return cs, cd


def test_decode_conventions_split_cut():
    cs, cd = _split_cut(0.5)
    vol = _volume([cs, cd])
    tilt = l2.build_tilt(vol, 0.5, n_gates=50)
    f = tilt.fields
    # value = (code - offset) / scale; below-threshold -> NaN; beyond data -> NaN
    np.testing.assert_allclose(f["DBZ"][5, 5], (cs["moments"]["REF"][0][5, 5] - 66.0) / 2.0)
    assert np.isnan(f["DBZ"][5, 0]) and np.isnan(f["DBZ"][5, 45])
    # legacy 8-bit ZDR keeps flag codes: background = -8.0
    assert f["ZDR"][5, 0] == pytest.approx(-8.0)
    # RF mask = VEL code 1 OR SW code 1; VEL/WIDTH NaN there
    assert tilt.range_folded_mask[0, 10] == 1 and tilt.range_folded_mask[1, 11] == 1
    assert tilt.range_folded_mask.sum() == 2
    assert np.isnan(f["VEL"][1, 11]) and np.isnan(f["WIDTH"][0, 10])
    assert tilt.surveillance.sweep.index == 0 and tilt.doppler.sweep.index == 1
    assert tilt.nyquist == 25.0


def test_zdr_16bit_background_is_nan():
    cs, cd = _split_cut(0.5, zdr_word=16)
    tilt = l2.build_tilt(_volume([cs, cd]), 0.5, n_gates=40)
    assert np.isnan(tilt.fields["ZDR"][5, 0])           # (0 - 418)/32 < -8 -> NaN
    finite = tilt.fields["ZDR"][np.isfinite(tilt.fields["ZDR"])]
    assert finite.min() >= l2.ZDR_MIN


def test_azimuth_floor_binning_and_1deg_upsampling():
    cs, cd = _split_cut(0.5)
    cs["az"] = _super_res_az(0.49)                     # radial at 0.49 -> bin 0, 359.99 -> bin 719
    vol = _volume([cs, cd])
    vals, _ = l2.decode_moment(vol, 0, "REF", 40)
    assert vals[0, 10] == pytest.approx((cs["moments"]["REF"][0][0, 10] - 66) / 2)
    # 1 degree sweeps fill both 0.5 degree bins
    bins, rep = l2._azimuth_bins(np.array([10.7, 11.2]), 360)
    assert rep == 2 and list(bins) == [20, 22]


def test_first_gate_offset_is_respected():
    cs, cd = _split_cut(0.5)
    codes, sc, off, _, w = cs["moments"]["REF"]
    cs["moments"]["REF"] = (codes, sc, off, 2375, w)   # first gate one gate further out
    vals, _ = l2.decode_moment(_volume([cs, cd]), 0, "REF", 45)
    assert np.isnan(vals[7, 0])
    assert vals[7, 4] == pytest.approx((codes[7, 3] - 66) / 2)


def test_cut_selection_latest_first_nearest_and_incomplete():
    cs0, cd0 = _split_cut(0.5)
    cs1, cd1 = _split_cut(0.9)
    cs2, cd2 = _split_cut(0.5)                           # SAILS repeat of 0.5 deg
    cd2["complete"] = False                              # still being received
    vol = _volume([cs0, cd0, cs1, cd1, cs2, cd2])
    t = l2.build_tilt(vol, 0.5, 40, selection="latest")
    assert (t.surveillance.sweep.index, t.doppler.sweep.index) == (4, 1)  # incomplete CD skipped
    t = l2.build_tilt(vol, 0.5, 40, selection="first")
    assert (t.surveillance.sweep.index, t.doppler.sweep.index) == (0, 1)
    t = l2.build_tilt(vol, 0.5, 40, selection="nearest", target_time=vol.time + datetime.timedelta(seconds=75))
    assert t.surveillance.sweep.index == 4
    with pytest.raises(ValueError):
        l2.build_tilt(vol, 1.8, 40)


# --------------------------------------------------------------------- geometry
def test_chip_geometry_matches_tornet_sample():
    # From TorNet file NUL_131118_050841_KLWX_485709s_N5.nc (verified gate-for-gate):
    # azimuth_limits=[-9, 51], range_limits=[120012, 180012] -> bins 702.., gate 471
    az_idx, g0 = chip_indices(-9.0, 120012.0)
    assert az_idx[0] == 702 and az_idx[18] == 0 and len(az_idx) == CHIP_N_AZ
    assert g0 == 471
    assert chip_limits(21.2, 150012.0) == (-9.0, 51.0, 120012.0, 180012.0)


def test_level3_index_alignment_and_upsampling():
    k1 = np.arange(360 * 1200, dtype=np.float32).reshape(360, 1200)
    grid = level3_to_tornet_grid(k1, 1300)
    assert grid.shape == (720, 1300)
    assert grid[2 * 17, 471] == k1[17, 471] == grid[2 * 17 + 1, 471]
    assert np.isnan(grid[0, 1250])


def test_netcdf_quantize_matches_netcdf4():
    x = np.array([0.5 + 51.54, 1.0517, -2.05, 7.9375], dtype=np.float32)
    np.testing.assert_allclose(netcdf_quantize(x, 1)[0], 52.0625)
    np.testing.assert_allclose(netcdf_quantize(x, 3)[1], round(1.0517 * 1024) / 1024)
    np.testing.assert_allclose(netcdf_quantize(x, 2)[2], -2.046875)
    netCDF4 = pytest.importorskip("netCDF4")
    import os, tempfile
    with tempfile.TemporaryDirectory() as d:
        fn = os.path.join(d, "q.nc")
        with netCDF4.Dataset(fn, "w") as ds:
            ds.createDimension("n", len(x))
            for lsd in (1, 2, 3):
                v = ds.createVariable(f"v{lsd}", "f4", ("n",), least_significant_digit=lsd)
                v[:] = x
        with netCDF4.Dataset(fn) as ds:
            for lsd in (1, 2, 3):
                np.testing.assert_array_equal(ds[f"v{lsd}"][:], netcdf_quantize(x, lsd))


def _fake_tornet_volume(n_gates=600):
    rng = np.random.default_rng(0)
    fields = {v: rng.normal(size=(720, n_gates, 2)).astype(np.float32) for v in ALL_VARIABLES}
    return TorNetVolume(site="KXXX", latitude=35.0, longitude=-97.0, vcp=212,
                        time=datetime.datetime(2022, 5, 1), elevations=np.array([0.5, 0.9]),
                        nyquist=np.array([25.0, 25.0], dtype=np.float32), fields=fields,
                        range_folded_mask=np.zeros((720, n_gates, 2), np.uint8), kdp_source="level3")


def test_extract_chip_layout_matches_read_file():
    vol = _fake_tornet_volume()
    c = extract_chip(vol, az_center=5.0, rng_center=40000.0)    # wraps through north
    for v in ALL_VARIABLES + ["range_folded_mask"]:
        assert c[v].shape == (1, CHIP_N_AZ, CHIP_N_RNG, 2)
    assert c["range_folded_mask"].dtype == np.float32
    az_idx, g0 = chip_indices(-25.0, 10000.0)
    np.testing.assert_array_equal(c["DBZ"][0], vol.fields["DBZ"][az_idx][:, g0:g0 + CHIP_N_RNG])
    for k in ("az_lower", "az_upper", "rng_lower", "rng_upper", "time", "label", "category"):
        assert c[k].shape == (1,)
    assert c["az_lower"][0] == -25.0 and c["rng_upper"][0] == 70000.0


def test_extract_chip_near_radar_and_beyond_data_is_nan():
    vol = _fake_tornet_volume(n_gates=300)
    c = extract_chip(vol, az_center=90.0, rng_center=20000.0)   # starts before first gate
    assert np.isnan(c["DBZ"][0, :, :40]).all() and np.isfinite(c["DBZ"][0, :, 60]).all()
    c = extract_chip(vol, az_center=90.0, rng_center=80000.0)   # runs past the last gate
    assert np.isnan(c["DBZ"][0, :, -10:]).all()


def test_stack_frames_and_tiles():
    vol = _fake_tornet_volume()
    frames = [extract_chip(vol, 100.0, 50000.0) for _ in range(4)]
    s = stack_frames(frames)
    assert s["DBZ"].shape == (4, CHIP_N_AZ, CHIP_N_RNG, 2) and s["az_lower"].shape == (1,)
    centers = tile_centers(vol, az_step=30, rng_step=30000)
    # 600 gates reach 152 km: rings centered at 30..180 km (last one partially filled)
    assert len(centers) == 12 * 6 and centers[0] == (0.0, 30000.0)


# --------------------------------------------------------------------- KDP fallback
def test_kdp_from_level2_linear_phase():
    n_g = 400
    rng_km = (2125 + 250 * np.arange(n_g)) / 1000.0
    phidp = np.tile(60.0 + 2.0 * rng_km, (720, 1)).astype(np.float32)   # 2 deg/km -> KDP 1
    dbz = np.full((720, n_g), 45.0, np.float32)
    dbz[:, 300:] = 10.0                                                   # light echo -> KDP 0
    rho = np.full((720, n_g), 0.98, np.float32)
    k = kdp_from_level2(dbz, rho, phidp)
    assert k.shape == (360, 1200)
    assert np.isnan(k[:, :8]).all()                                       # Level-III bin = L2 gate + 8
    np.testing.assert_allclose(k[:, 8 + 50:8 + 250], 1.0, atol=0.051)
    np.testing.assert_allclose(k[:, 8 + 310:8 + 390], 0.0)


# --------------------------------------------------------------------- dealiasing helpers
def test_dealias_split_merge_and_float16_correction():
    from tornet.nexrad import dealias
    x = np.arange(720 * 3, dtype=np.float32).reshape(720, 3)
    np.testing.assert_array_equal(dealias._merge_super_res(dealias._split_super_res(x)), x)

    class _Const:
        """Stand-in model: predicts +1 fold everywhere (category 4)."""
        def predict_on_batch(self, x):
            out = np.zeros(x.shape[:3] + (6,), np.float32)
            out[..., 4] = 1.0
            return out

    vel = np.full((720, 64), 0.5, np.float32)
    vel[3, 3] = np.nan
    out = dealias.UNetVDA(_Const())(vel, 25.77)
    # float16 arithmetic as in TorNet: 0.5 + 2*fp16(25.77) -> 52.03125, stored as 52.0
    assert out[0, 0] == pytest.approx(52.03125)
    assert netcdf_quantize(out, 1)[0, 0] == 52.0
    assert np.isnan(out[3, 3])


# --------------------------------------------------------------------- model-ready inputs
def test_volume_to_model_inputs_matches_training_preprocessing():
    from tornet.data import preprocess as pp
    from tornet.nexrad.model_inputs import MODEL_INPUT_KEYS, ModelInputs, volume_to_model_inputs

    vol = _fake_tornet_volume()
    centers = [(5.0, 40000.0), (200.0, 90000.0), (300.0, 140000.0)]
    mi = volume_to_model_inputs(vol, centers=centers)
    assert len(mi) == 3 and set(mi.inputs) == set(MODEL_INPUT_KEYS)
    for k in MODEL_INPUT_KEYS:
        assert mi.inputs[k].shape == (3, CHIP_N_AZ, CHIP_N_RNG, 2) and mi.inputs[k].dtype == np.float32
    # identical to the training pipeline: read_file sample -> remove_time_dim -> add_coordinates
    ref = extract_chip(vol, *centers[1])
    pp.remove_time_dim(ref)
    pp.add_coordinates(ref, include_az=False, tilt_last=True, backend=np)
    for k in MODEL_INPUT_KEYS:
        np.testing.assert_array_equal(mi.inputs[k][1], ref[k])
    assert mi.chips["az_lower"][0] == -25.0 and mi.chips["rng_center"][2] == 140000.0
    assert len(list(mi.batches(2))) == 2

    import os, tempfile
    with tempfile.TemporaryDirectory() as d:
        fn = os.path.join(d, "mi.npz")
        mi.save(fn)
        m2 = ModelInputs.load(fn)
    assert m2.site == "KXXX" and m2.time == vol.time and m2.kdp_source == "level3"
    for k in MODEL_INPUT_KEYS:
        np.testing.assert_array_equal(m2.inputs[k], mi.inputs[k])
    # full-sweep tiling by default; include_az adds a third coordinate channel
    assert len(volume_to_model_inputs(vol, az_step=30, rng_step=30000)) == 72
    assert volume_to_model_inputs(vol, centers=centers[:1], include_az=True).inputs["coordinates"].shape[-1] == 3
