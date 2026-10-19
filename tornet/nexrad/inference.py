"""
Run a TorNet model over a full radar sweep by tiling it into TorNet-sized chips.

The CNN baseline ends in a GlobalMaxPooling over a per-pixel "heatmap" layer, so each
chip yields a chip-level probability and a heatmap.  Heatmaps of overlapping chips are
max-combined onto the full [720 azimuth, n_gates] polar grid.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tornet.nexrad.model_inputs import chips_to_model_inputs
from tornet.nexrad.pipeline import (CHIP_N_AZ, CHIP_N_RNG, TorNetVolume, chip_indices,
                                    extract_chip, tile_centers)

DEFAULT_MODEL_REPO = "tornet-ml/tornado_detector_baseline_v1"
DEFAULT_MODEL_FILE = "tornado_detector_baseline.keras"


def load_tornet_model(model_path: Optional[str] = None):
    import keras
    from tornet.models.keras.layers import CoordConv2D, FillNaNs

    if model_path is None:
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(repo_id=DEFAULT_MODEL_REPO, filename=DEFAULT_MODEL_FILE)
    return keras.saving.load_model(model_path, compile=False,
                                   custom_objects={"CoordConv2D": CoordConv2D, "FillNaNs": FillNaNs})


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class SweepPredictor:
    def __init__(self, model, batch_size: int = 32):
        import keras

        self.model = model
        self.input_keys = list(model.input.keys())
        self.batch_size = batch_size
        heat = model.get_layer("heatmap").output
        self.heatmap_model = keras.Model(inputs=model.inputs, outputs=heat)

    def predict(self, volume: TorNetVolume,
                centers: Optional[List[Tuple[float, float]]] = None,
                **tile_kwargs):
        """
        Returns (detections, prob_map) where
          detections: list of dicts {az_center, rng_center, az_lower, rng_lower, prob}
          prob_map:   [720, n_gates] max tornado probability from chip heatmaps
        """
        centers = centers if centers is not None else tile_centers(volume, **tile_kwargs)
        prob_map = np.zeros((720, volume.n_gates), dtype=np.float32)
        detections = []
        for i in range(0, len(centers), self.batch_size):
            cs = centers[i:i + self.batch_size]
            chips = [extract_chip(volume, a, r) for a, r in cs]
            x = chips_to_model_inputs(chips, keys=self.input_keys)
            heat = np.asarray(self.heatmap_model.predict_on_batch(x))  # ~10% faster than predict()
            if heat.ndim == 3:
                heat = heat[..., None]
            prob = _sigmoid(heat[..., 0].reshape(len(chips), -1).max(axis=1))
            for (a, r), c, p, h in zip(cs, chips, prob, heat[..., 0]):
                detections.append({"az_center": a, "rng_center": r,
                                   "az_lower": float(c["az_lower"][0]), "rng_lower": float(c["rng_lower"][0]),
                                   "prob": float(p)})
                self._paste(prob_map, c, _sigmoid(h))
        return detections, prob_map

    @staticmethod
    def _paste(prob_map: np.ndarray, chip: Dict[str, np.ndarray], heat: np.ndarray):
        """Nearest-upsample a chip heatmap to chip resolution and max it into the sweep grid."""
        ha, hr = heat.shape
        up = np.repeat(np.repeat(heat, int(np.ceil(CHIP_N_AZ / ha)), 0), int(np.ceil(CHIP_N_RNG / hr)), 1)
        up = up[:CHIP_N_AZ, :CHIP_N_RNG]
        az_idx, g0 = chip_indices(float(chip["az_lower"][0]), float(chip["rng_lower"][0]))
        G = prob_map.shape[1]
        lo, hi = max(g0, 0), min(g0 + CHIP_N_RNG, G)
        if hi <= lo:
            return
        sub = prob_map[az_idx, lo:hi]
        prob_map[az_idx, lo:hi] = np.maximum(sub, up[:, lo - g0:hi - g0])
