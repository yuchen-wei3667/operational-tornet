"""
Level-II-only KDP that emulates the Level-III (ORPG) KDP product used in TorNet.

Use this only when the Level-III N0K/NAK products are unavailable.  It is an
approximation of the ORPG dual-pol preprocessor, NOT a bit-exact reproduction:

1. recombine 0.5-deg radials into 1-deg radials (as the ORPG does for DP products)
2. process phase on gates with RHOHV >= 0.85; output where Z is finite and
   (RHOHV >= 0.92 or Z >= 20 dBZ), matching where Level-III KDP is valid;
   KDP = 0 where Z < 25 dBZ, as in Level-III
3. unwrap, gap-fill and median-filter PHIDP along each radial
4. KDP = 1/2 d(PHIDP)/dr from a least-squares fit over 13 gates (Z >= 40 dBZ)
   or 33 gates (Z < 40 dBZ); ORPG-like, window lengths tuned against Level-III
5. quantize to the Level-III data levels and place on the Level-III range-bin grid,
   so it goes through exactly the same TorNet alignment as real Level-III data
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve1d, median_filter

from tornet.nexrad.level2 import FIRST_GATE_M, GATE_M
from tornet.nexrad.level3 import LEVEL3_TO_LEVEL2_GATE_OFFSET

L3_KDP_SCALE = 20.0
L3_KDP_OFFSET = 43.0
L3_KDP_MIN = (2 - L3_KDP_OFFSET) / L3_KDP_SCALE     # code 2
L3_KDP_MAX = (255 - L3_KDP_OFFSET) / L3_KDP_SCALE   # code 255
L3_N_BINS = 1200


def _recombine_1deg(x: np.ndarray, circular_deg: bool = False) -> np.ndarray:
    a, b = x[0::2], x[1::2]
    if circular_deg:
        ca = np.exp(1j * np.deg2rad(a))
        cb = np.exp(1j * np.deg2rad(b))
        s = np.where(np.isfinite(a), ca, 0) + np.where(np.isfinite(b), cb, 0)
        out = np.mod(np.rad2deg(np.angle(s)), 360.0)
        out[~(np.isfinite(a) | np.isfinite(b))] = np.nan
        return out.astype(np.float32)
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.stack([a, b]), axis=0).astype(np.float32)


def _ls_slope_kernel(window: int) -> np.ndarray:
    k = np.arange(window, dtype=np.float64) - (window - 1) / 2.0
    return (k / np.sum(k ** 2))[::-1]


def kdp_from_level2(dbz: np.ndarray,
                    rhohv: np.ndarray,
                    phidp: np.ndarray,
                    phase_rho_min: float = 0.85,
                    out_rho_min: float = 0.92,
                    out_z_min: float = 20.0,
                    short_window: int = 13,
                    long_window: int = 33,
                    z_short_dbz: float = 40.0,
                    z_zero_dbz: float = 25.0,
                    median_window: int = 9) -> np.ndarray:
    """
    dbz, rhohv, phidp: [720, n_gates] fields of one tilt on the TorNet grid
    Returns KDP [360, 1200] on the Level-III bin grid (deg/km, NaN missing)

    Output gates follow the Level-III validity pattern, fitted on Level-III products:
    finite Z and (RHOHV >= out_rho_min or Z >= out_z_min)  (IoU ~0.85).
    """
    z1 = _recombine_1deg(dbz)
    r1 = _recombine_1deg(rhohv)
    p1 = _recombine_1deg(phidp, circular_deg=True)
    n_az, n_g = z1.shape

    with np.errstate(invalid="ignore"):
        phase_ok = np.isfinite(z1) & np.isfinite(p1) & (r1 >= phase_rho_min)
        out_ok = np.isfinite(z1) & ((r1 >= out_rho_min) | (z1 >= out_z_min))

    # Unwrap / gap-fill / median-filter PHIDP along range
    x = np.arange(n_g)
    phi = np.full((n_az, n_g), np.nan, dtype=np.float64)
    for i in range(n_az):
        good = phase_ok[i]
        if good.sum() < 3:
            continue
        p = np.rad2deg(np.unwrap(np.deg2rad(p1[i, good])))
        xg = x[good]
        filled = np.interp(x, xg, p)
        filled[: xg[0]] = np.nan
        filled[xg[-1] + 1:] = np.nan
        phi[i] = filled
    valid = np.isfinite(phi)
    phi_f = phi
    if median_window > 1:
        phi_f = median_filter(np.where(valid, phi, 0.0), size=(1, median_window), mode="nearest")

    gate_km = GATE_M / 1000.0
    filled = np.where(valid, phi_f, 0.0)
    kdp_short = 0.5 * convolve1d(filled, _ls_slope_kernel(short_window), axis=1, mode="nearest") / gate_km
    kdp_long = 0.5 * convolve1d(filled, _ls_slope_kernel(long_window), axis=1, mode="nearest") / gate_km
    # A window touching the end of the phase data falls back to the shorter window
    sup_l = convolve1d(valid.astype(np.float32), np.ones(long_window), axis=1, mode="constant") >= long_window
    sup_s = convolve1d(valid.astype(np.float32), np.ones(short_window), axis=1, mode="constant") >= short_window
    with np.errstate(invalid="ignore"):
        use_short = (z1 >= z_short_dbz) | ~sup_l
    kdp = np.where(use_short, kdp_short, kdp_long)
    kdp = np.where(sup_s | sup_l, kdp, 0.0)   # too little phase data: no differential phase signal
    with np.errstate(invalid="ignore"):
        kdp[z1 < z_zero_dbz] = 0.0            # Level-III KDP is 0 in light echo
    kdp[~out_ok] = np.nan

    # Quantize to Level-III data levels
    kdp = np.clip(np.round(kdp * L3_KDP_SCALE) / L3_KDP_SCALE, L3_KDP_MIN, L3_KDP_MAX)

    # Level-II gate g -> Level-III bin g - LEVEL3_TO_LEVEL2_GATE_OFFSET
    out = np.full((n_az, L3_N_BINS), np.nan, dtype=np.float32)
    shift = -LEVEL3_TO_LEVEL2_GATE_OFFSET
    n = min(n_g, L3_N_BINS - shift)
    out[:, shift:shift + n] = kdp[:, :n]
    return out
