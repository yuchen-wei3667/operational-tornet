"""
NEXRAD Level-II decoding onto the native TorNet polar grid.

These conventions were derived from the TorNet paper and verified gate-for-gate
against released TorNet samples and the matching Level-II archives:

* Grid:      0.5 deg x 250 m native "super-resolution" polar grid.  No interpolation.
             Radial with azimuth ``az`` goes to bin ``floor(az / 0.5)`` (720 bins).
             Gate index ``g`` is the Level-II gate centered at 2125 + 250*g m.
             1-deg (360 radial) sweeps are nearest-neighbor upsampled to 720 bins.
* Tilts:     the 0.5 and 0.9 deg elevation cuts.  For split cuts, DBZ/ZDR/RHOHV (and
             PHIDP) come from the surveillance (CS) cut and VEL/WIDTH from the Doppler
             (CD) cut.  Each cut is chosen independently as the one nearest in time to
             the frame time (SAILS/MESO-SAILS repeats are just more candidates).
* Decoding:  value = (code - offset) / scale.  Codes 0 (below threshold) and
             1 (range folded) -> NaN, except ZDR, where all codes are decoded and
             values < -8 dB are NaN (so legacy 8-bit ZDR has a -8.0 dB background,
             16-bit ZDR a NaN background, as in TorNet).
* RF mask:   ``range_folded_mask`` = 1 where the Doppler-cut velocity OR spectrum
             width code is 1; VEL and WIDTH are NaN there.
"""
from __future__ import annotations

import bz2
import datetime
import gzip
import io
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

N_AZ = 720               # 0.5 deg radials
AZ_RES = 0.5
FIRST_GATE_M = 2125.0    # range of the center of Level-II gate 0
GATE_M = 250.0
TORNET_TILTS = (0.5, 0.9)
ZDR_MIN = -8.0

# Level-II moment names -> TorNet variable names
SURVEILLANCE_MOMENTS = {"REF": "DBZ", "ZDR": "ZDR", "RHO": "RHOHV", "PHI": "PHIDP"}
DOPPLER_MOMENTS = {"VEL": "VEL", "SW": "WIDTH"}
CUT_SELECTIONS = ("latest", "first", "nearest")

FileLike = Union[str, bytes, io.IOBase]


@dataclass
class SweepInfo:
    index: int
    elevation: float          # VCP target elevation angle (deg)
    moments: Tuple[str, ...]
    time: datetime.datetime   # time of first radial
    nyquist: float
    n_radials: int
    complete: bool = True     # last radial flagged end-of-elevation / end-of-volume

    @property
    def is_surveillance(self) -> bool:
        return "REF" in self.moments and ("ZDR" in self.moments or "RHO" in self.moments)

    @property
    def is_doppler(self) -> bool:
        return "VEL" in self.moments


@dataclass
class Level2Volume:
    site: str
    latitude: float
    longitude: float
    vcp: int
    time: datetime.datetime           # volume start time
    sweeps: List[SweepInfo]
    _file: object = field(repr=False, default=None)


@dataclass
class Cut:
    volume: Level2Volume
    sweep: SweepInfo


@dataclass
class PolarTilt:
    """One TorNet tilt on the native 720 x n_gates grid (NaN = missing)."""
    elevation: float
    fields: Dict[str, np.ndarray]     # DBZ, ZDR, RHOHV, PHIDP, VEL (aliased), WIDTH
    range_folded_mask: np.ndarray      # uint8
    nyquist: float
    surveillance: Cut
    doppler: Cut

    @property
    def time(self) -> datetime.datetime:
        return self.surveillance.sweep.time


def _read_bytes(source: FileLike) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    elif isinstance(source, io.IOBase):
        data = source.read()
    else:
        with open(source, "rb") as f:
            data = f.read()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    elif data[:3] == b"BZh":
        data = bz2.decompress(data)
    return data


def open_level2(source: Union[FileLike, Level2Volume]) -> Level2Volume:
    """
    Opens a Level-II Archive II file (raw, .gz, or a concatenation of real-time
    chunks, possibly an incomplete volume) without building a full Py-ART Radar.
    """
    if isinstance(source, Level2Volume):
        return source
    from pyart.io.nexrad_level2 import NEXRADLevel2File

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = NEXRADLevel2File(io.BytesIO(_read_bytes(source)))

    sweeps = []
    for i in range(f.nscans):
        msgs = f.scan_msgs[i]
        if len(msgs) == 0:
            continue
        rec = f.radial_records[msgs[0]]
        moments = tuple(k for k in ("REF", "VEL", "SW", "ZDR", "PHI", "RHO") if k in rec)
        t0, dt = f.get_times([i])
        # Message 31 radial status (Py-ART calls this byte 'radial_spacing'):
        # 2 = end of elevation, 4 = end of volume.  Real-time volumes end mid-sweep.
        last = f.radial_records[msgs[-1]].get("msg_header", {})
        status = last.get("radial_status", last.get("radial_spacing"))
        sweeps.append(SweepInfo(
            index=i,
            elevation=float(f.get_target_angles([i])[0]),
            moments=moments,
            time=t0 + datetime.timedelta(seconds=float(dt[0])),
            nyquist=float(f.get_nyquist_vel([i])[0]),
            n_radials=len(msgs),
            complete=status in (2, 4) if status is not None else True,
        ))

    lat, lon, _ = f.location()
    vh = getattr(f, "volume_header", {}) or {}
    site = vh.get("icao", b"")
    site = site.decode(errors="ignore") if isinstance(site, bytes) else str(site)
    vcp = f.get_vcp_pattern()
    return Level2Volume(site=site.strip("\x00 "), latitude=float(lat), longitude=float(lon),
                        vcp=int(vcp) if vcp is not None else -1,
                        time=sweeps[0].time if sweeps else None, sweeps=sweeps, _file=f)


def find_cuts(volumes: Sequence[Level2Volume], elevation: float,
              tol: float = 0.25) -> Tuple[List[Cut], List[Cut]]:
    """All complete (surveillance, Doppler) cuts within ``tol`` deg of ``elevation``."""
    surv, dopp = [], []
    for vol in volumes:
        for s in vol.sweeps:
            if abs(s.elevation - elevation) > tol or not s.complete:
                continue
            if s.is_surveillance:
                surv.append(Cut(vol, s))
            if s.is_doppler:
                dopp.append(Cut(vol, s))
    return surv, dopp


def choose_cut(cuts: Sequence[Cut], selection: str = "latest",
               target_time: Optional[datetime.datetime] = None) -> Cut:
    if not cuts:
        raise ValueError("no candidate cuts")
    if selection == "latest":
        return max(cuts, key=lambda c: c.sweep.time)
    if selection == "first":
        return min(cuts, key=lambda c: c.sweep.time)
    if selection == "nearest":
        if target_time is None:
            raise ValueError("selection='nearest' requires target_time")
        return min(cuts, key=lambda c: abs(c.sweep.time - target_time))
    raise ValueError(f"selection must be one of {CUT_SELECTIONS}")


def _azimuth_bins(az: np.ndarray, n_radials: int) -> Tuple[np.ndarray, int]:
    """Returns (bin index per radial, repeat factor) on a 720 bin grid."""
    az = np.mod(np.asarray(az, dtype=np.float64), 360.0)
    if n_radials > 400:   # super-resolution 0.5 deg
        return np.floor(az / AZ_RES).astype(int) % N_AZ, 1
    return (np.floor(az).astype(int) % 360) * 2, 2


def decode_moment(volume: Level2Volume,
                  sweep: int,
                  moment: str,
                  n_gates: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decodes one moment of one sweep onto the [720, n_gates] TorNet grid.
    Returns (values float32 with NaN for missing, raw codes int32 with -1 for no data)
    """
    f = volume._file
    msgs = f.scan_msgs[sweep]
    rec = f.radial_records[msgs[0]][moment]
    ngates_m = int(rec["ngates"])
    if int(rec["gate_spacing"]) != int(GATE_M):
        raise ValueError(f"Unsupported gate spacing {rec['gate_spacing']} for {moment}")
    raw = f.get_data(moment, ngates_m, scans=[sweep], raw_data=True)
    raw = np.asarray(np.ma.filled(raw, 0), dtype=np.int32)
    scale, offset = float(rec["scale"]), float(rec["offset"])

    # Place on the common gate grid (gate 0 centered at 2125 m)
    shift = int(round((float(rec["first_gate"]) - FIRST_GATE_M) / GATE_M))
    codes = np.full((raw.shape[0], n_gates), -1, dtype=np.int32)
    lo, hi = max(shift, 0), min(shift + ngates_m, n_gates)
    if hi > lo:
        codes[:, lo:hi] = raw[:, lo - shift:hi - shift]

    values = (codes.astype(np.float32) - np.float32(offset)) / np.float32(scale)
    if moment == "ZDR":
        with np.errstate(invalid="ignore"):
            values[(codes < 0) | (values < ZDR_MIN)] = np.nan
    else:
        values[codes <= 1] = np.nan

    bins, rep = _azimuth_bins(f.get_azimuth_angles([sweep]), len(msgs))
    out = np.full((N_AZ, n_gates), np.nan, dtype=np.float32)
    out_codes = np.full((N_AZ, n_gates), -1, dtype=np.int32)
    for r in range(rep):  # numpy assignment: later radials win on duplicate bins
        out[bins + r] = values
        out_codes[bins + r] = codes
    return out, out_codes


def build_tilt(volumes: Union[Level2Volume, Sequence[Level2Volume]],
               elevation: float,
               n_gates: int,
               selection: str = "latest",
               target_time: Optional[datetime.datetime] = None,
               tol: float = 0.25) -> PolarTilt:
    """Decodes all TorNet variables (except KDP) for one tilt."""
    if isinstance(volumes, Level2Volume):
        volumes = [volumes]
    surv, dopp = find_cuts(volumes, elevation, tol)
    if not surv or not dopp:
        avail = sorted({round(s.elevation, 2) for v in volumes for s in v.sweeps})
        raise ValueError(f"No complete {elevation} deg cut (surveillance + Doppler); available: {avail}")
    cs = choose_cut(surv, selection, target_time)
    cd = choose_cut(dopp, selection, target_time)

    fields: Dict[str, np.ndarray] = {}
    for mom, name in SURVEILLANCE_MOMENTS.items():
        if mom in cs.sweep.moments:
            fields[name], _ = decode_moment(cs.volume, cs.sweep.index, mom, n_gates)
    missing = [v for v in ("DBZ", "ZDR", "RHOHV") if v not in fields]
    if missing:
        raise ValueError(f"Surveillance cut is missing moments {missing} at {elevation} deg")
    fields["VEL"], vcodes = decode_moment(cd.volume, cd.sweep.index, "VEL", n_gates)
    fields["WIDTH"], wcodes = decode_moment(cd.volume, cd.sweep.index, "SW", n_gates)
    rf = (vcodes == 1) | (wcodes == 1)
    fields["VEL"][rf] = np.nan
    fields["WIDTH"][rf] = np.nan
    return PolarTilt(elevation=round(cs.sweep.elevation, 1), fields=fields,
                     range_folded_mask=rf.astype(np.uint8), nyquist=cd.sweep.nyquist,
                     surveillance=cs, doppler=cd)
