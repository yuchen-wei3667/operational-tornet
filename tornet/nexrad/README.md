# NEXRAD Level-II → TorNet model inputs (`tornet.nexrad`)

This package converts NEXRAD Level-II data (archived or live) into exactly the
tensors the TorNet models were trained on, following Veillette et al. (2024/2025),
*A Benchmark Dataset for Tornado Detection and Prediction using Full-Resolution
Polarimetric Weather Radar Data* (AIES).

Every rule below was checked gate-for-gate against released TorNet samples and the
matching NOAA archives (`scripts/nexrad/validate_against_tornet.py`). Where the paper
text and the released data disagree, the data wins. Those cases are marked **(data)**.

## How TorNet samples were built

| Item | Rule |
|---|---|
| Grid | Native polar 0.5° × 250 m "super-resolution" grid; **no interpolation**. Radial azimuth `az` → bin `floor(az/0.5)` (720 bins). Gate `g` = Level-II gate centered at 2125 + 250·g m. |
| Chip | 120 radials × 240 gates = 60° × **60 km** (paper says 80 km) **(data)**. `azimuth_limits` are integers (center ± 30°, may be negative, e.g. `[-9, 51]`). First gate index `g0 = floor((range_limits[0] − 2125)/250)`. |
| Tilts | 0.5° and 0.9° cuts. For split cuts, DBZ/ZDR/RHOHV come from the surveillance (CS) cut and VEL/WIDTH from the Doppler (CD) cut. |
| Decoding | `(code − offset)/scale`. Codes 0/1 (below threshold / range folded) → NaN. |
| ZDR | All codes decoded, values < −8 dB → NaN **(data)**. For legacy 8-bit ZDR (2013-era) the background is therefore −8.0. For 16-bit ZDR (later) it is NaN. |
| Range-folded mask | 1 where the Doppler-cut **VEL or spectrum-width** code is 1; VEL and WIDTH are NaN there **(data)**. |
| VEL | Dealiased with U-Net VDA ([mit-ll/unet-vda](https://github.com/mit-ll/unet-vda)) on the full 360° sweep. Corrections were computed in **float16** (`vel + k·2·fp16(nyquist)`) **(data)**. |
| KDP | From **Level-III** product 163 (`N0K` for 0.5°, `NAK` for 0.9°), 1° radials nearest-upsampled to 0.5°. Level-III bin *k* is placed at Level-II gate *k*. Level-III bins start at 0 km, so **TorNet KDP sits ~2.1 km (8.5 gates) farther out than the other fields** **(data)**. This module reproduces that shift. |
| Storage | Values quantized like netCDF4 `least_significant_digit` (DBZ 1, VEL 1, KDP 2, RHOHV 3, ZDR 3, WIDTH 2). |
| Frames | Each cut is taken from the cut nearest in time to the frame time ("nearest available data for each variable"), so SAILS/MESO-SAILS repeats are candidates. The last frame matched exactly with this rule. Earlier frames show irregular ±6 min jitter in which cut TorNet picked. Every individual frame is still reproducible exactly from *some* cut. |

## Measured fidelity

Last frame of each sample (what the baseline CNN uses). Score = fraction of all
2×120×240 gates identical to the TorNet file (NaN == NaN), with Level-III KDP:

| Variable | Exact match |
|---|---|
| DBZ, RHOHV, ZDR, WIDTH, KDP, range_folded_mask | 100% (bit-exact) |
| VEL (dealiased) | ≈99.9% (remaining gates: U-Net fold decisions at class boundaries; different TF build/hardware) |

See the validation summary in the commit / PR for the exact sample set.

**Level-II-only KDP fallback** (`kdp_source="level2"`, used when Level-III is not
available): an ORPG-like estimate tuned against Level-III products. Against Level-III
it gives MAE ≈ 0.08 °/km, correlation ≈ 0.8, valid-mask IoU ≈ 0.85. It is an
approximation, not the training distribution, so prefer Level-III.

## Usage

```python
from tornet.nexrad import level2_to_tornet_volume, extract_chip, iter_chips
from tornet.nexrad.dealias import UNetVDA
from tornet.nexrad import fetch

vda = UNetVDA.from_pretrained()          # downloads + converts the unet-vda weights once

# Archived volume (+ Level-III KDP fetched from AWS by cut time)
vol = level2_to_tornet_volume("KTLX20240507_000047_V06",
                              level3_kdp=fetch.level3_kdp_fetcher,
                              dealiaser=vda)
sample = extract_chip(vol, az_center=215.0, rng_center=74_000.0)
# `sample` has the same keys/shapes as tornet.data.loader.read_file(f, n_frames=1)

# Live data: newest volume from the real-time chunk feed (usable mid-volume)
data, complete = fetch.realtime_volume_bytes("KTLX")
vol = level2_to_tornet_volume(data, level3_kdp=fetch.level3_kdp_fetcher, dealiaser=vda)
```

Full-sweep inference, real-time loop:

```bash
python scripts/nexrad/run_realtime_tornet.py --sites KTLX KFWS --out-dir ./tornet_rt
python scripts/nexrad/run_realtime_tornet.py --level2-file KTLX20240507_000047_V06 --out-dir .
```

The runner polls `unidata-nexrad-level2-chunks`. As soon as new complete 0.5°/0.9°
cuts exist (~1–2 min into a volume), it builds inputs. Cuts not yet received come
from the previous volume. It waits up to `--level3-wait` s for Level-III KDP, then
falls back to the Level-II estimate (recorded as `kdp_source` in the output). It tiles
the sweep with overlapping TorNet chips and writes per-chip probabilities (JSON) and a
max-combined probability map on the polar grid (`.npz`).

Reproduce TorNet samples and measure fidelity:

```bash
python scripts/nexrad/validate_against_tornet.py $TORNET_ROOT/train/2022/*.nc \
    --max-files 50 --last-frame-only --kdp-source both
```

## Production notes

* **Dependencies:** `arm_pyart` (Level-II/III decoding), `scipy`, `keras>=3` with any
  backend. `tensorflow` is needed once, to convert the unet-vda checkpoint
  (`tornet.nexrad.dealias.download_unet_vda_weights`). Ship the resulting `.npz` and
  point `TORNET_UNET_VDA_WEIGHTS` at it.
* **Latency (CPU):** decode + dealias both tilts ≈ 3–5 s. Tiling a 230 km sweep is
  ~380 chips; on CPU, together with model load, this took ~30 s.
* **VCPs:** requires a 0.5° and 0.9° cut (VCP 12, 212, 215, 35). VCPs without a 0.9°
  cut (31, 32) raise `ValueError`. TorNet contains no such data.
* **Range:** chips reaching beyond the data, or inside 2.125 km, are NaN-filled.
  TorNet does the same at its edges, and the model maps NaN to the background flag.
* **Data sources:** `unidata-nexrad-level2`, `unidata-nexrad-level2-chunks`,
  `unidata-nexrad-level3` (AWS Open Data, anonymous HTTPS). Level-III on AWS covers
  2020 and later. Earlier Level-III comes from `gcp-public-data-nexrad-l3` daily tars;
  see the validation script.
