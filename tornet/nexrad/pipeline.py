"""
Converts NEXRAD Level-II (+ optionally Level-III KDP) into TorNet model inputs.

    vol = level2_to_tornet_volume("KTLX20240507_000047_V06",
                                  level3_kdp=["TLX_N0K_...", "TLX_NAK_..."],
                                  dealiaser=UNetVDA.from_pretrained())
    sample = extract_chip(vol, az_center=215.3, rng_center=74_000.)  # read_file() layout
    for sample in iter_chips(vol): ...                               # full-sweep tiling

The resulting dicts have exactly the keys/shapes/dtypes of
``tornet.data.loader.read_file(f, n_frames=1)`` so the existing preprocessing and
models (``tornet.data.preprocess``, ``tornet.models.keras``) apply unchanged.
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np

from tornet.data.constants import ALL_VARIABLES
from tornet.nexrad import level2 as l2
from tornet.nexrad.kdp import kdp_from_level2
from tornet.nexrad.level3 import KDP_PRODUCT_FOR_TILT, decode_level3_kdp, level3_to_tornet_grid

logger = logging.getLogger(__name__)

CHIP_N_AZ = 120          # 60 deg of 0.5 deg radials
CHIP_N_RNG = 240         # 60 km of 250 m gates
CHIP_AZ_HALF_WIDTH = 30  # deg
CHIP_RNG_HALF_WIDTH = 30000.0  # m

# least_significant_digit used when TorNet netCDF files were written.  netCDF4
# quantizes values to a power-of-two precision, which we reproduce for bit-exactness.
TORNET_LEAST_SIGNIFICANT_DIGIT = {"DBZ": 1, "VEL": 1, "KDP": 2, "RHOHV": 3, "ZDR": 3, "WIDTH": 2}

KDP_SOURCES = ("level3", "level2", "auto")


def netcdf_quantize(data: np.ndarray, least_significant_digit: int) -> np.ndarray:
    """Same quantization netCDF4-python applies with ``least_significant_digit``."""
    precision = 10.0 ** -least_significant_digit
    exp = np.log10(precision)
    exp = int(np.floor(exp)) if exp < 0 else int(np.ceil(exp))
    bits = np.ceil(np.log2(10.0 ** -exp))
    scale = 2.0 ** bits
    return (np.around(scale * data) / scale).astype(np.float32)


@dataclass
class TorNetVolume:
    """Full-sweep TorNet variables, [720 azimuth, n_gates, n_tilts] per variable."""
    site: str
    latitude: float
    longitude: float
    vcp: int
    time: datetime.datetime
    elevations: np.ndarray
    nyquist: np.ndarray
    fields: Dict[str, np.ndarray]
    range_folded_mask: np.ndarray
    kdp_source: str
    sweep_times: List[datetime.datetime] = field(default_factory=list)

    @property
    def n_gates(self) -> int:
        return self.range_folded_mask.shape[1]


Level3Source = Union[l2.FileLike, np.ndarray, None]
# callable(site, product, cut_time) -> Level-III bytes or None
Level3Fetcher = Callable[[str, str, datetime.datetime], Optional[bytes]]


def level2_to_tornet_volume(
    level2_source: Union[l2.FileLike, l2.Level2Volume, Sequence[Union[l2.FileLike, l2.Level2Volume]]],
    level3_kdp: Optional[Union[Sequence[Level3Source], Level3Fetcher]] = None,
    kdp_source: str = "auto",
    dealiaser: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    tilts: Sequence[float] = l2.TORNET_TILTS,
    selection: str = "latest",
    target_time: Optional[datetime.datetime] = None,
    n_gates: Optional[int] = None,
    quantize: bool = True,
) -> TorNetVolume:
    """
    level2_source:  Level-II file path / bytes (raw, gz, or concatenated real-time chunks)
                    or a list of them (e.g. consecutive volumes) to choose cuts from.
    level3_kdp:     Level-III KDP per tilt (N0K, NAK) as paths/bytes/decoded [360, n]
                    arrays (entries may be None), or a callable
                    ``(site, product, surveillance_cut_time) -> bytes | None`` such as
                    ``tornet.nexrad.fetch.level3_kdp_fetcher``.
    kdp_source:     'level3' (require Level-III, exact TorNet match),
                    'level2' (always estimate from Level-II PHIDP; approximate),
                    'auto'   (Level-III when available, otherwise Level-II estimate)
    dealiaser:      callable(vel[720, n], nyquist) -> dealiased vel.  TorNet used
                    U-Net VDA (``tornet.nexrad.dealias.UNetVDA``).  None = no dealiasing.
    selection:      which cut to use per tilt when several exist (SAILS, several volumes):
                    'latest' (real-time default), 'first', or 'nearest' to target_time
                    (how TorNet frames were built).
    n_gates:        size of range dimension (default: longest decoded moment)
    quantize:       reproduce TorNet's netCDF storage quantization
    """
    if kdp_source not in KDP_SOURCES:
        raise ValueError(f"kdp_source must be one of {KDP_SOURCES}")
    if isinstance(level2_source, (str, bytes, bytearray, l2.Level2Volume)) or hasattr(level2_source, "read"):
        level2_source = [level2_source]
    volumes = [l2.open_level2(s) for s in level2_source]
    if n_gates is None:
        n_gates = max(_max_gates(v) for v in volumes)
    if level3_kdp is None:
        level3_kdp = [None] * len(tilts)
    if not callable(level3_kdp) and len(level3_kdp) != len(tilts):
        raise ValueError("level3_kdp must have one entry per tilt")

    per_tilt: Dict[str, List[np.ndarray]] = {v: [] for v in ALL_VARIABLES}
    rf, elevs, nyqs, times, kdp_used = [], [], [], [], []
    for t, elev in enumerate(tilts):
        tilt = l2.build_tilt(volumes, elev, n_gates, selection=selection, target_time=target_time)
        f = tilt.fields
        vel = f["VEL"]
        if dealiaser is not None:
            # only run over the velocity extent (Doppler cuts are shorter than REF)
            cols = np.flatnonzero(np.isfinite(vel).any(axis=0))
            n_vel = int(cols[-1]) + 1 if cols.size else 0
            vel = vel.copy()
            if n_vel:
                vel[:, :n_vel] = dealiaser(vel[:, :n_vel], tilt.nyquist)
            vel = np.where(np.isfinite(f["VEL"]), vel, np.nan).astype(np.float32)

        if callable(level3_kdp):
            l3 = None
            if kdp_source != "level2":
                l3 = level3_kdp(tilt.surveillance.volume.site, KDP_PRODUCT_FOR_TILT[t], tilt.time)
        else:
            l3 = level3_kdp[t]
        if l3 is not None and kdp_source != "level2":
            k1 = l3 if isinstance(l3, np.ndarray) else decode_level3_kdp(l3)
            kdp_used.append("level3")
        elif kdp_source == "level3":
            raise ValueError(f"kdp_source='level3' but no Level-III KDP available for tilt {elev}")
        else:
            if "PHIDP" not in f:
                raise ValueError("Level-II volume has no PHIDP; cannot estimate KDP")
            k1 = kdp_from_level2(f["DBZ"], f["RHOHV"], f["PHIDP"])
            kdp_used.append("level2")
        kdp = level3_to_tornet_grid(k1, n_gates)

        for name, arr in (("DBZ", f["DBZ"]), ("VEL", vel), ("KDP", kdp), ("RHOHV", f["RHOHV"]),
                          ("ZDR", f["ZDR"]), ("WIDTH", f["WIDTH"])):
            per_tilt[name].append(netcdf_quantize(arr, TORNET_LEAST_SIGNIFICANT_DIGIT[name]) if quantize else arr)
        rf.append(tilt.range_folded_mask)
        elevs.append(tilt.elevation)
        nyqs.append(tilt.nyquist)
        times.append(tilt.time)

    ref = volumes[0]
    return TorNetVolume(
        site=ref.site, latitude=ref.latitude, longitude=ref.longitude, vcp=ref.vcp,
        time=target_time or max(times),
        elevations=np.array(elevs, dtype=np.float64), nyquist=np.array(nyqs, dtype=np.float32),
        fields={k: np.stack(v, axis=-1).astype(np.float32) for k, v in per_tilt.items()},
        range_folded_mask=np.stack(rf, axis=-1), kdp_source=",".join(kdp_used), sweep_times=times,
    )


def _max_gates(vol: l2.Level2Volume) -> int:
    f = vol._file
    n = 0
    for s in vol.sweeps[:8]:
        rec = f.radial_records[f.scan_msgs[s.index][0]]
        for m in s.moments:
            n = max(n, int(rec[m]["ngates"]))
    return n


def chip_limits(az_center: float, rng_center: float):
    """
    TorNet chip limits for a chip centered at (az_center deg, rng_center m).
    azimuth limits are integers (as in TorNet, e.g. [-9, 51]); range limits are
    center -/+ 30 km.
    """
    az_lower = float(int(np.round(az_center)) - CHIP_AZ_HALF_WIDTH)
    rng_lower = float(rng_center - CHIP_RNG_HALF_WIDTH)
    return az_lower, az_lower + 2 * CHIP_AZ_HALF_WIDTH, rng_lower, rng_lower + 2 * CHIP_RNG_HALF_WIDTH


def chip_indices(az_lower: float, rng_lower: float):
    """(azimuth bin indices [120], first gate index) used by TorNet for given limits."""
    a0 = int(np.floor(np.mod(az_lower, 360.0) / l2.AZ_RES))
    az_idx = (a0 + np.arange(CHIP_N_AZ)) % l2.N_AZ
    g0 = int(np.floor((rng_lower - l2.FIRST_GATE_M) / l2.GATE_M))
    return az_idx, g0


def _take_gates(arr: np.ndarray, az_idx: np.ndarray, g0: int, fill):
    """arr [720, G, T] -> [120, 240, T] with out-of-range gates filled."""
    G = arr.shape[1]
    out = np.full((CHIP_N_AZ, CHIP_N_RNG) + arr.shape[2:], fill, dtype=arr.dtype)
    lo, hi = max(g0, 0), min(g0 + CHIP_N_RNG, G)
    if hi > lo:
        out[:, lo - g0:hi - g0] = arr[az_idx, lo:hi]
    return out


def extract_chip(volume: TorNetVolume,
                 az_center: Optional[float] = None,
                 rng_center: Optional[float] = None,
                 az_lower: Optional[float] = None,
                 rng_lower: Optional[float] = None,
                 variables: Sequence[str] = ALL_VARIABLES) -> Dict[str, np.ndarray]:
    """
    Extracts one TorNet sample (single frame) in the layout of
    ``tornet.data.loader.read_file(..., n_frames=1, tilt_last=True)``.

    Specify either the center (az_center deg, rng_center m) or explicit TorNet limits
    (az_lower, rng_lower), e.g. those stored in a TorNet file / catalog.
    """
    if az_lower is None or rng_lower is None:
        if az_center is None or rng_center is None:
            raise ValueError("Provide (az_center, rng_center) or (az_lower, rng_lower)")
        az_lower, az_upper, rng_lower, rng_upper = chip_limits(az_center, rng_center)
    else:
        az_upper = az_lower + 2 * CHIP_AZ_HALF_WIDTH
        rng_upper = rng_lower + 2 * CHIP_RNG_HALF_WIDTH
    az_idx, g0 = chip_indices(az_lower, rng_lower)

    out: Dict[str, np.ndarray] = {}
    for v in variables:
        out[v] = _take_gates(volume.fields[v], az_idx, g0, np.nan)[None]
    out["range_folded_mask"] = _take_gates(volume.range_folded_mask, az_idx, g0, 0)[None].astype(np.float32)
    t = int((volume.time - datetime.datetime(1970, 1, 1)).total_seconds()) if volume.time else 0
    out.update({
        "label": np.array([0], dtype=np.uint8),
        "category": np.array([1], dtype=np.int64),
        "event_id": np.array([0], dtype=np.int64),
        "ef_number": np.array([-1], dtype=np.int64),
        "az_lower": np.array([az_lower], dtype=np.float32),
        "az_upper": np.array([az_upper], dtype=np.float32),
        "rng_lower": np.array([rng_lower], dtype=np.float32),
        "rng_upper": np.array([rng_upper], dtype=np.float32),
        "time": np.array([t], dtype=np.int64),
        "tornado_start_time": np.array([0], dtype=np.int64),
        "tornado_end_time": np.array([0], dtype=np.int64),
    })
    return out


def tile_centers(volume: TorNetVolume,
                 az_step: float = 15.0,
                 rng_step: float = 15000.0,
                 rng_min: float = 0.0,
                 rng_max: Optional[float] = None):
    """
    Chip centers that tile the full sweep with overlap (defaults: 50% overlap).
    Chip centers lie on gate centers so chip limits match TorNet's convention.
    Returns list of (az_center, rng_center).
    """
    if rng_max is None:
        rng_max = l2.FIRST_GATE_M + l2.GATE_M * volume.n_gates
    centers = []
    r = max(rng_min, 0.0) + CHIP_RNG_HALF_WIDTH
    while r - CHIP_RNG_HALF_WIDTH < rng_max:
        for a in np.arange(0.0, 360.0, az_step):
            centers.append((float(a), float(r)))
        r += rng_step
    return centers


def iter_chips(volume: TorNetVolume, centers=None, **tile_kwargs) -> Iterator[Dict[str, np.ndarray]]:
    for az_c, rng_c in (centers if centers is not None else tile_centers(volume, **tile_kwargs)):
        yield extract_chip(volume, az_c, rng_c)


def stack_frames(frames: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Stacks single-frame samples (oldest first, same chip limits) into a multi-frame
    sample [T, az, rng, tilt], matching ``read_file(..., n_frames=T)``.
    """
    out = dict(frames[-1])
    for k in list(ALL_VARIABLES) + ["range_folded_mask", "time", "label"]:
        if k in out:
            out[k] = np.concatenate([f[k] for f in frames], axis=0)
    return out
