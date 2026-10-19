"""
NEXRAD Level-III digital specific differential phase (product 163; N0K / NAK ...)

TorNet took KDP from Level-III (1 deg x 250 m) and nearest-neighbor upsampled it
to 0.5 deg.  Verified against TorNet samples:

* Level-III radial with start angle ``a`` fills 0.5-deg bins 2*round(a), 2*round(a)+1
* Level-III range bin ``k`` is placed at Level-II gate index ``k`` (index aligned).
  Level-III bins start at 0 km while Level-II gate 0 is centered at 2.125 km, so
  TorNet KDP is physically displaced ~2.1 km (8.5 gates) outward relative to the
  other variables.  This module reproduces that alignment exactly.
* value = (code - offset) / scale (scale=20, offset=43 for product 163), codes 0/1 -> NaN

Level-III product names: N0K = lowest (0.5 deg) cut, NAK = second cut (0.9 deg in
VCPs 12/212/215), N1K = third, ...
"""
from __future__ import annotations

import io
import struct
import warnings
from typing import Union

import numpy as np

from tornet.nexrad.level2 import N_AZ, _read_bytes

KDP_PRODUCT_CODE = 163
# Level-III product mnemonic for each TorNet tilt (lowest and second-lowest cut)
KDP_PRODUCT_FOR_TILT = {0: "N0K", 1: "NAK"}
# Level-III bin k <-> Level-II gate k + LEVEL3_TO_LEVEL2_GATE_OFFSET is the *physical*
# match (bin centers 125 + 250k m vs 2125 + 250g m).  Used by the Level-II KDP fallback
# to produce data on the Level-III bin grid.
LEVEL3_TO_LEVEL2_GATE_OFFSET = -8


def decode_level3_kdp(source: Union[str, bytes, io.IOBase]) -> np.ndarray:
    """
    Decodes a Level-III KDP product onto its native [360, n_bins] grid (deg/km, NaN missing).
    Row r holds the radial whose start azimuth rounds to r degrees.
    """
    from pyart.io.nexrad_level3 import NEXRADLevel3File

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = NEXRADLevel3File(io.BytesIO(_read_bytes(source)))
    code = f.prod_descr["product_code"]
    if code != KDP_PRODUCT_CODE:
        raise ValueError(f"Expected product {KDP_PRODUCT_CODE} (KDP), got {code}")
    if f.packet_header.get("first_bin", 0) != 0:
        raise ValueError("Unexpected first_bin != 0 in Level-III KDP product")
    scale, offset = struct.unpack(">ff", f.prod_descr["threshold_data"][:8])
    raw = np.asarray(f.raw_data, dtype=np.float32)
    vals = np.where(raw <= 1, np.nan, (raw - np.float32(offset)) / np.float32(scale)).astype(np.float32)
    out = np.full((360, raw.shape[1]), np.nan, dtype=np.float32)
    rows = np.round(np.mod(f.get_azimuth(), 360.0)).astype(int) % 360
    out[rows] = vals
    return out


def level3_to_tornet_grid(kdp_1deg: np.ndarray, n_gates: int) -> np.ndarray:
    """
    [360, n_bins] Level-III grid -> [720, n_gates] TorNet grid
    (nearest-neighbor in azimuth, index-aligned in range as in TorNet).
    """
    out = np.full((N_AZ, n_gates), np.nan, dtype=np.float32)
    n = min(n_gates, kdp_1deg.shape[1])
    up = np.repeat(kdp_1deg[:, :n], 2, axis=0)
    out[:, :n] = up
    return out
