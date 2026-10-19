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

Held-out check: 27 TorNet samples (21 from 2022, 6 from 2013; 27 radars; TOR/WRN/NUL
categories), last frame, Level-III KDP. Score = fraction of all 2×120×240 gates that
are identical to the TorNet file (NaN == NaN):

| Variable | Mean exact | Samples 100% exact |
|---|---|---|
| DBZ, RHOHV, ZDR | 98.0–98.2% | 25 / 27 |
| WIDTH, range_folded_mask | 98.7% / 99.85% | 22 / 27, 24 / 27 |
| KDP (Level-III) | 97.4% | 22 / 25 |
| VEL (dealiased) | 98.3% (99.9% on the 25) | n/a (U-Net decisions at class boundaries) |

All shortfalls come from **which cut** TorNet assigned to a frame, not from gate
processing. For every outlier, `frame_diag`-style search over neighbouring cuts finds
a cut that reproduces the TorNet chip at **100%**. TorNet sometimes took a cut 1–6
minutes after the frame time (e.g. NAK from the next volume). This jitter does not
follow a reproducible rule. In real time the inputs are the latest cuts, so it does
not apply.

**Level-II-only KDP fallback** (`kdp_source="level2"`, used when Level-III is not
available): an ORPG-like estimate tuned against Level-III products.
* Full sweeps, vs. Level-III: MAE ≈ 0.08 °/km, correlation ≈ 0.8, valid-mask IoU ≈ 0.85.
* The 27 held-out TorNet chips: MAE 0.13 °/km, correlation 0.64, valid-mask IoU 0.89.

It is an approximation of the training distribution, so prefer Level-III. Level-III
is published in real time on AWS, typically within a minute or two.

## Usage

### Level-II → model-ready inputs in one call

```python
from tornet.nexrad import ModelInputs, fetch, level2_to_model_inputs

mi = level2_to_model_inputs("KTLX20240507_000047_V06",          # path, bytes, .gz, or live chunks
                            level3_kdp=fetch.level3_kdp_fetcher)  # omit for Level-II-only KDP
mi.inputs        # {DBZ, VEL, KDP, RHOHV, ZDR, WIDTH, range_folded_mask, coordinates}: float32 [N,120,240,2]
mi.chips         # az_center, rng_center, az/rng_lower/upper per chip: [N]
logits = model.predict(mi.inputs)                              # nothing else to do
mi.save("KTLX_inputs.npz"); ModelInputs.load("KTLX_inputs.npz")
```

By default the whole sweep is tiled with overlapping chips (`az_step`, `rng_step`,
`rng_max`). Pass `centers=[(az_deg, range_m), ...]` for specific chips. The tensors are
exactly what the TorNet training pipeline feeds the model (`read_file` →
`remove_time_dim` → `add_coordinates(include_az=False)`): physical units with NaNs,
because normalization and NaN filling happen inside the model. Use `include_az=True`
for models trained with azimuth coordinates. Dealiasing uses U-Net VDA by default
(`dealiaser=None` to skip).

The same thing from the command line, writing one `.npz` per scan:

```bash
python scripts/nexrad/level2_to_model_inputs.py KTLX20240507_000047_V06 --out-dir inputs/
python scripts/nexrad/level2_to_model_inputs.py KTLX20240507_000047_V06 --no-level3 --center 215 74
python scripts/nexrad/level2_to_model_inputs.py --site KTLX --out-dir inputs/   # newest live scan
```

### Lower-level API

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
* **Latency:** see *Benchmark* below. On 4 CPU cores a full 230 km sweep takes ~5 s to
  convert and ~18 s end to end with inference. For full sweeps, stream chips with
  `iter_model_input_batches` (or `SweepPredictor`). Building all ~384 chips as one array
  is ~700 MB, and the first touch of that memory alone costs ~4.5 s.
* **VCPs:** requires a 0.5° and 0.9° cut (VCP 12, 212, 215, 35). VCPs without a 0.9°
  cut (31, 32) raise `ValueError`. TorNet contains no such data.
* **Range:** chips reaching beyond the data, or inside 2.125 km, are NaN-filled.
  TorNet does the same at its edges, and the model maps NaN to the background flag.
* **Data sources:** `unidata-nexrad-level2`, `unidata-nexrad-level2-chunks`,
  `unidata-nexrad-level3` (AWS Open Data, anonymous HTTPS). Level-III on AWS covers
  2020 and later. Earlier Level-III comes from `gcp-public-data-nexrad-l3` daily tars;
  see the validation script.

## Benchmark

`scripts/nexrad/benchmark_level2.py` measures full-sweep runs (230 km, 384 chips,
1832 gates). Setup: 4-core cloud VM, no GPU, TensorFlow 2.21 / Keras 3.15. Data:
17 volumes from 17 radars (13 from 2022, 4 from 2013, KTLX 2024), 5–19 MB each, VCPs
12/212/215. Figures are the median (p90) of warm runs; files are read from memory.

| Stage | KDP from Level-II | KDP from Level-III |
|---|---|---|
| Level-II decode (Py-ART) | 1.97 s (3.75) | 2.47 s (3.51) |
| Cut selection + moment decode | 0.19 s | 0.18 s |
| U-Net VDA dealiasing, 2 tilts | 1.90 s (2.51) | 1.80 s (2.38) |
| KDP | 0.42 s (estimate) | 0.02 s (decode) |
| Quantization | 0.09 s | 0.09 s |
| Chips + coordinates, streamed | 0.47 s | 0.44 s |
| **Level-II → model-ready inputs** | **5.3 s (7.0)** | **4.9 s (6.3)** |
| TorNet inference, 384 chips | 13.2 s (13.9) | 12.7 s (13.8) |
| **End to end** | **18.6 s (20.2)** | **17.8 s (19.4)** |

* **One-time startup:** loading U-Net VDA takes 3.8 s and the TorNet model 1.0 s.
  Fetching Level-III from AWS adds 1.8 s (p90 2.1 s) per volume. Peak RSS is 2.9 GB.
* **Decode time** grows with file size: 0.7 s for 2013 files, 2.1 s for 2022, 4.0 s
  for the 2024 KTLX volume.
* **Inference** dominates the total (~33 ms per chip on CPU). `predict_on_batch` is
  used because it is ~10% faster than `predict`. To go faster, use a GPU or tile more
  coarsely. For example, `az_step=30, rng_step=30_000` gives no overlap and about a
  quarter of the chips.
* **Volume cadence:** a full volume arrives every 4–6 min (VCP 212/215), and new
  0.5° cuts come every 1–2 min with SAILS. So one CPU process keeps up with a few
  radars; scale out by radar.

```bash
python scripts/nexrad/benchmark_level2.py data/*_V06 --repeat 2 --kdp-source both --infer \
    --json-out bench.json
```
