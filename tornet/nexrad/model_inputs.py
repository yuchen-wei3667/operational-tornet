"""
NEXRAD Level-II scans -> model-ready TorNet input tensors in one call.

    from tornet.nexrad.model_inputs import level2_to_model_inputs
    mi = level2_to_model_inputs("KTLX20240507_000047_V06")          # whole sweep, tiled
    mi = level2_to_model_inputs(data, centers=[(215.0, 74_000.0)])  # chosen chips
    probs = model.predict(mi.inputs)                                 # no further processing

``mi.inputs`` is a dict of float32 arrays [N, 120, 240, 2] keyed like the inputs of
the TorNet Keras models (DBZ, VEL, KDP, RHOHV, ZDR, WIDTH, range_folded_mask,
coordinates).  This is exactly what the TorNet training pipeline feeds the model
(``read_file`` -> ``remove_time_dim`` -> ``add_coordinates``): raw physical values with
NaNs, since normalization and NaN filling happen inside the model.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from tornet.data import preprocess as pp
from tornet.data.constants import ALL_VARIABLES
from tornet.nexrad.pipeline import TorNetVolume, extract_chip, level2_to_tornet_volume, tile_centers

MODEL_INPUT_KEYS = tuple(ALL_VARIABLES) + ("range_folded_mask", "coordinates")
CHIP_META_KEYS = ("az_lower", "az_upper", "rng_lower", "rng_upper")

_DEFAULT_DEALIASER = None


def default_dealiaser():
    """U-Net VDA as used for TorNet, loaded once per process."""
    global _DEFAULT_DEALIASER
    if _DEFAULT_DEALIASER is None:
        from tornet.nexrad.dealias import UNetVDA
        _DEFAULT_DEALIASER = UNetVDA.from_pretrained()
    return _DEFAULT_DEALIASER


def chip_to_model_input(chip: Dict[str, np.ndarray], include_az: bool = False,
                        keys: Sequence[str] = MODEL_INPUT_KEYS) -> Dict[str, np.ndarray]:
    """One ``extract_chip``/``read_file`` sample -> unbatched model input [120, 240, C]."""
    s = dict(chip)                      # shallow: the steps below only rebind keys
    pp.remove_time_dim(s)
    pp.add_coordinates(s, include_az=include_az, tilt_last=True, backend=np)
    return {k: s[k].astype(np.float32, copy=False) for k in keys}


def chips_to_model_inputs(chips: Sequence[Dict[str, np.ndarray]], include_az: bool = False,
                          keys: Sequence[str] = MODEL_INPUT_KEYS) -> Dict[str, np.ndarray]:
    """Single-frame chips -> batched model inputs [N, 120, 240, C]."""
    out: Dict[str, np.ndarray] = {}
    for i, c in enumerate(chips):
        x = chip_to_model_input(c, include_az, keys)
        if not out:                     # preallocate once; avoids np.stack's extra copy
            out = {k: np.empty((len(chips),) + x[k].shape, np.float32) for k in keys}
        for k in keys:
            out[k][i] = x[k]
    return out


@dataclass
class ModelInputs:
    """Batched, model-ready inputs for N chips of one radar scan."""
    inputs: Dict[str, np.ndarray]           # key -> float32 [N, 120, 240, C]
    chips: Dict[str, np.ndarray]            # az_center, rng_center, az/rng lower/upper -> [N]
    site: str = ""
    time: Optional[datetime.datetime] = None
    vcp: int = 0
    latitude: float = np.nan
    longitude: float = np.nan
    elevations: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sweep_times: List[datetime.datetime] = field(default_factory=list)
    kdp_source: str = ""

    def __len__(self) -> int:
        return len(self.chips["az_center"])

    def batches(self, batch_size: int = 32) -> Iterator[Dict[str, np.ndarray]]:
        """Yields input dicts of at most ``batch_size`` chips (views, no copies)."""
        for i in range(0, len(self), batch_size):
            yield {k: v[i:i + batch_size] for k, v in self.inputs.items()}

    def save(self, path: str, compress: bool = True):
        """Writes one .npz holding the inputs (keys as-is) and chip/scan metadata."""
        meta = {
            "site": np.array(self.site), "vcp": np.array(self.vcp),
            "time": np.array(self.time.isoformat() if self.time else ""),
            "latitude": np.array(self.latitude), "longitude": np.array(self.longitude),
            "elevations": np.asarray(self.elevations),
            "sweep_times": np.array([t.isoformat() for t in self.sweep_times]),
            "kdp_source": np.array(self.kdp_source),
        }
        arrays = dict(self.inputs)
        arrays.update({f"chip_{k}": v for k, v in self.chips.items()})
        arrays.update({f"meta_{k}": v for k, v in meta.items()})
        (np.savez_compressed if compress else np.savez)(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "ModelInputs":
        """Reads a file written by ``save``."""
        with np.load(path) as z:
            inputs = {k: z[k] for k in z.files if not k.startswith(("chip_", "meta_"))}
            chips = {k[5:]: z[k] for k in z.files if k.startswith("chip_")}
            m = {k[5:]: z[k] for k in z.files if k.startswith("meta_")}
        t = str(m["time"])
        return cls(inputs=inputs, chips=chips, site=str(m["site"]), vcp=int(m["vcp"]),
                   time=datetime.datetime.fromisoformat(t) if t else None,
                   latitude=float(m["latitude"]), longitude=float(m["longitude"]),
                   elevations=m["elevations"],
                   sweep_times=[datetime.datetime.fromisoformat(str(s)) for s in m["sweep_times"]],
                   kdp_source=str(m["kdp_source"]))


def volume_to_model_inputs(volume: TorNetVolume,
                           centers: Optional[Sequence[Tuple[float, float]]] = None,
                           include_az: bool = False,
                           keys: Sequence[str] = MODEL_INPUT_KEYS,
                           **tile_kwargs) -> ModelInputs:
    """
    Full-sweep TorNet volume -> model inputs for the chips centered at ``centers``
    [(azimuth deg, range m), ...], or for an overlapping tiling of the whole sweep
    (``tile_centers`` kwargs: az_step, rng_step [m], rng_min, rng_max).
    """
    centers = list(centers) if centers is not None else tile_centers(volume, **tile_kwargs)
    meta = {"az_center": np.array([c[0] for c in centers], np.float32),
            "rng_center": np.array([c[1] for c in centers], np.float32)}
    meta.update({k: np.empty(len(centers), np.float32) for k in CHIP_META_KEYS})
    variables = [k for k in keys if k in ALL_VARIABLES]
    inputs: Dict[str, np.ndarray] = {}
    for i, (a, r) in enumerate(centers):
        chip = extract_chip(volume, a, r, variables=variables)
        x = chip_to_model_input(chip, include_az, keys)
        if not inputs:
            inputs = {k: np.empty((len(centers),) + x[k].shape, np.float32) for k in keys}
        for k in keys:
            inputs[k][i] = x[k]
        for k in CHIP_META_KEYS:
            meta[k][i] = chip[k][0]
    if not inputs:
        inputs = {k: np.zeros((0, 120, 240, 2), np.float32) for k in keys}
    return ModelInputs(inputs=inputs, chips=meta, site=volume.site, time=volume.time,
                       vcp=volume.vcp, latitude=volume.latitude, longitude=volume.longitude,
                       elevations=volume.elevations, sweep_times=list(volume.sweep_times),
                       kdp_source=volume.kdp_source)


def level2_to_model_inputs(level2_source,
                           centers: Optional[Sequence[Tuple[float, float]]] = None,
                           level3_kdp=None,
                           kdp_source: str = "auto",
                           dealiaser: Union[str, None, object] = "default",
                           selection: str = "latest",
                           target_time: Optional[datetime.datetime] = None,
                           include_az: bool = False,
                           keys: Sequence[str] = MODEL_INPUT_KEYS,
                           **tile_kwargs) -> ModelInputs:
    """
    Level-II scan(s) -> model-ready TorNet inputs.

    level2_source: Level-II file path / bytes (raw, gz, bz2, or concatenated real-time
                   chunks) or a list of consecutive volumes.
    centers:       chip centers [(az deg, range m)]; None tiles the whole sweep.
    level3_kdp:    Level-III KDP (list per tilt or fetcher callable, e.g.
                   ``tornet.nexrad.fetch.level3_kdp_fetcher``).  None + kdp_source='auto'
                   estimates KDP from Level-II alone.
    dealiaser:     'default' = U-Net VDA (as TorNet), None = no dealiasing, or a callable.
    """
    if isinstance(dealiaser, str):
        if dealiaser != "default":
            raise ValueError("dealiaser must be 'default', None or a callable")
        dealiaser = default_dealiaser()
    volume = level2_to_tornet_volume(level2_source, level3_kdp=level3_kdp, kdp_source=kdp_source,
                                     dealiaser=dealiaser, selection=selection, target_time=target_time)
    return volume_to_model_inputs(volume, centers=centers, include_az=include_az, keys=keys,
                                  **tile_kwargs)


def iter_model_input_batches(volume: TorNetVolume,
                             batch_size: int = 64,
                             centers: Optional[Sequence[Tuple[float, float]]] = None,
                             include_az: bool = False,
                             keys: Sequence[str] = MODEL_INPUT_KEYS,
                             **tile_kwargs) -> Iterator[ModelInputs]:
    """
    Like ``volume_to_model_inputs`` but yields batches of at most ``batch_size`` chips,
    so a full sweep (~400 chips, ~700 MB as one array) never has to be held at once.
    """
    centers = list(centers) if centers is not None else tile_centers(volume, **tile_kwargs)
    for i in range(0, len(centers), batch_size):
        yield volume_to_model_inputs(volume, centers=centers[i:i + batch_size],
                                     include_az=include_az, keys=keys)
