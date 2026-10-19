"""
Velocity dealiasing used to build TorNet.

TorNet dealiased Level-II velocity with the U-Net velocity dealiasing algorithm
(U-Net VDA) of Veillette et al. (2023), "A deep learning-based velocity dealiasing
algorithm derived from the WSR-88D Open Radar Product Generator", AIES.
Code and pretrained weights: https://github.com/mit-ll/unet-vda

The original release is a TF 2.9 ``tf.keras`` SavedModel which no longer loads in
current TensorFlow/Keras 3.  This module re-implements the identical architecture
in Keras 3 (so it runs on any keras backend) and loads the original weights,
converted once to a backend-independent ``.npz`` file.

Usage:
    dealiaser = UNetVDA.from_pretrained()          # downloads/converts if needed
    vel_dealiased = dealiaser(vel_720xN, nyquist)   # full 360 deg sweep
"""
from __future__ import annotations

import os
import re
import urllib.request
from typing import List, Optional

import numpy as np

UNET_VDA_REPO_RAW = "https://github.com/mit-ll/unet-vda/raw/main/models/dealias_sn16_csi9764.SavedModel/variables/"
UNET_VDA_CKPT_FILES = ("variables.index", "variables.data-00000-of-00001")
DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "tornet")
DEFAULT_WEIGHTS_NAME = "unet_vda_sn16_csi9764.npz"

# Must match feature_extraction.py of unet-vda
_POOL = [(1, 2), (2, 4), (4, 2), (4, 4)]
_START_NEURONS = 16
_N_OUTPUTS = 6
# Total downsampling factor of the U-Net along (azimuth, range)
AZ_MULTIPLE = 32
RANGE_MULTIPLE = 64
# Periodic azimuth padding used by unet-vda (degrees == radials at 1 deg)
AZ_PAD = 12
BAD_FILL = -3.0  # normalized fill value for missing velocity
N_RADIALS_1DEG = 360


def build_unet_vda_model():
    """
    Builds the U-Net VDA segmentation network in Keras 3.

    Input:  [batch, n_az, n_rng, 1]  velocity normalized by nyquist, missing = -3
    Output: [batch, n_az, n_rng, 6]  logits for fold categories (-2,-1,0,0,+1,+2)

    Returns (model, layers_in_checkpoint_order)
    """
    import keras

    ordered: List = []  # layers in the order their variables appear in the checkpoint

    def conv(filters, kernel):
        layer = keras.layers.Conv2D(filters, kernel, padding="same")
        ordered.append(layer)
        return layer

    def bn():
        layer = keras.layers.BatchNormalization()
        ordered.append(layer)
        return layer

    def polar_block(x, input_channels, kernel_size, pool_size, output_channels, alpha=0.1):
        conv_a = conv(output_channels, kernel_size)
        conv_b = conv(output_channels, kernel_size)
        conv_skip = conv(output_channels, 1) if input_channels != output_channels else None
        x1 = conv_skip(x) if conv_skip is not None else x
        x1 = keras.layers.AveragePooling2D(pool_size)(x1)
        x2 = keras.layers.LeakyReLU(alpha)(conv_a(x))
        x2 = keras.layers.LeakyReLU(alpha)(conv_b(x2))
        x2 = keras.layers.AveragePooling2D(pool_size)(x2)
        return keras.layers.Add()([x1, x2])

    def up_block(x, output_channels, kernel_size, upsample_factor, x_skip=None):
        if x_skip is not None:
            x = keras.layers.Concatenate(axis=-1)([x_skip, x])
        x = keras.layers.ReLU()(bn()(x))
        x = keras.layers.UpSampling2D(size=upsample_factor)(x)
        x = conv(output_channels, kernel_size)(x)
        x = keras.layers.ReLU()(bn()(x))
        x = conv(output_channels, kernel_size)(x)
        x = conv(output_channels, kernel_size)(x)
        return x

    sn = _START_NEURONS
    inp = keras.Input(shape=(None, None, 1), name="vel_normalized")
    x0 = polar_block(inp, 1, (7, 7), _POOL[0], sn)
    x1 = polar_block(x0, sn, (5, 5), _POOL[1], 2 * sn)
    x2 = polar_block(x1, 2 * sn, (3, 3), _POOL[2], 4 * sn)
    x3 = polar_block(x2, 4 * sn, (3, 3), _POOL[3], 8 * sn)

    x = keras.layers.Concatenate()([conv(sn * 8, 3)(x3), conv(sn * 8, 5)(x3), conv(sn * 8, 7)(x3)])
    x = up_block(x, sn * 8, (3, 3), _POOL[3])
    x = up_block(x, sn * 4, (3, 3), _POOL[2], x_skip=x2)
    x = up_block(x, sn * 2, (3, 3), _POOL[1], x_skip=x1)
    x = up_block(x, sn, (3, 3), (1, 2), x_skip=x0)
    out = conv(_N_OUTPUTS, (3, 3))(x)

    model = keras.Model(inputs=inp, outputs=out, name="unet_vda")
    return model, ordered


def convert_unet_vda_checkpoint(ckpt_prefix: str, out_npz: str) -> str:
    """
    Converts the original unet-vda TF checkpoint
    (``.../dealias_sn16_csi9764.SavedModel/variables/variables``) into an .npz of
    weight arrays in checkpoint order.  Requires tensorflow (only for this step).
    """
    import tensorflow as tf

    reader = tf.train.load_checkpoint(ckpt_prefix)
    # Only model variables ("variables/<n>/.ATTRIBUTES/VARIABLE_VALUE"), not optimizer slots
    names = [k for k in reader.get_variable_to_shape_map()
             if re.fullmatch(r"variables/\d+/\.ATTRIBUTES/VARIABLE_VALUE", k)]
    names = sorted(names, key=lambda k: int(k.split("/")[1]))
    arrays = {f"w{i:03d}": reader.get_tensor(k) for i, k in enumerate(names)}
    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
    np.savez(out_npz, **arrays)
    return out_npz


def download_unet_vda_weights(cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    """
    Downloads the original unet-vda checkpoint from GitHub and converts it to .npz.
    Returns path of the .npz file.  Requires tensorflow for the one-time conversion;
    production images should run this once at build time and ship the .npz.
    """
    out_npz = os.path.join(cache_dir, DEFAULT_WEIGHTS_NAME)
    if os.path.exists(out_npz):
        return out_npz
    ckpt_dir = os.path.join(cache_dir, "unet_vda_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    for fname in UNET_VDA_CKPT_FILES:
        dest = os.path.join(ckpt_dir, fname)
        if not os.path.exists(dest):
            urllib.request.urlretrieve(UNET_VDA_REPO_RAW + fname, dest)
    return convert_unet_vda_checkpoint(os.path.join(ckpt_dir, "variables"), out_npz)


def load_unet_vda_weights(model_layers: List, weights_npz: str) -> None:
    with np.load(weights_npz) as f:
        arrays = [f[k] for k in sorted(f.files)]
    i = 0
    for layer in model_layers:
        n = len(layer.weights)
        layer.set_weights(arrays[i:i + n])
        i += n
    if i != len(arrays):
        raise ValueError(f"Weight file {weights_npz} has {len(arrays)} arrays, model consumed {i}")


def _split_super_res(vel: np.ndarray) -> np.ndarray:
    """[720, R] -> [2, 360, R]  (even radials, odd radials)"""
    return np.stack([vel[0::2], vel[1::2]], axis=0)


def _merge_super_res(vel2: np.ndarray) -> np.ndarray:
    """[2, 360, R] -> [720, R]"""
    out = np.empty((2 * vel2.shape[1], vel2.shape[2]), dtype=vel2.dtype)
    out[0::2] = vel2[0]
    out[1::2] = vel2[1]
    return out


class UNetVDA:
    """
    Applies U-Net VDA to a full 360-degree velocity sweep, following the
    preprocessing of the unet-vda reference notebook:

      1. 0.5-deg (720 radial) data is split into two interleaved 1-deg arrays
      2. each array is padded 12 deg on each side with periodic boundary conditions
      3. velocity is normalized by nyquist, missing values filled with -3
      4. the network predicts the number of folds; vel += n * 2 * nyquist
    """

    def __init__(self, model, float16_correction: bool = True):
        self.model = model
        self.float16_correction = float16_correction

    @classmethod
    def from_pretrained(cls, weights_npz: Optional[str] = None, cache_dir: str = DEFAULT_CACHE_DIR):
        if weights_npz is None:
            weights_npz = os.environ.get("TORNET_UNET_VDA_WEIGHTS") or download_unet_vda_weights(cache_dir)
        model, ordered = build_unet_vda_model()
        load_unet_vda_weights(ordered, weights_npz)
        return cls(model)

    def predict_folds(self, vel_norm_batch: np.ndarray) -> np.ndarray:
        """vel_norm_batch: [B, n_az, n_rng, 1] normalized.  Returns argmax category [B,n_az,n_rng]"""
        logits = self.model.predict(vel_norm_batch.astype(np.float32), verbose=0)
        logits = np.clip(np.asarray(logits), -50.0, 50.0)
        return np.argmax(logits, axis=-1)

    def __call__(self, vel: np.ndarray, nyquist: float) -> np.ndarray:
        """
        vel:     [n_az, n_rng] aliased velocity (m/s), NaN for missing.
                 n_az must be 720 (super-res) or 360.
        nyquist: nyquist velocity of the sweep (m/s)
        Returns dealiased velocity with same shape (NaN preserved).
        """
        vel = np.asarray(vel, dtype=np.float32)
        n_az, n_rng = vel.shape
        if n_az == 2 * N_RADIALS_1DEG:
            arrays = _split_super_res(vel)
        elif n_az == N_RADIALS_1DEG:
            arrays = vel[None]
        else:
            raise ValueError(f"Expected 360 or 720 radials, got {n_az}")

        # Pad range to a multiple of the network's downsampling factor
        r_pad = (-n_rng) % RANGE_MULTIPLE
        x = np.pad(arrays, ((0, 0), (0, 0), (0, r_pad)), constant_values=np.nan)
        # Periodic azimuth padding
        x = np.concatenate([x[:, -AZ_PAD:], x, x[:, :AZ_PAD]], axis=1)
        xn = x / np.float32(nyquist)
        xn = np.where(np.isfinite(xn), xn, BAD_FILL).astype(np.float32)

        cat = self.predict_folds(xn[..., None])  # [B, 384, R]
        cat = cat[:, AZ_PAD:AZ_PAD + N_RADIALS_1DEG, :n_rng]

        folds = np.zeros(cat.shape, dtype=np.float32)
        folds[cat == 1] = -2.0
        folds[cat == 2] = -1.0
        folds[cat == 4] = 1.0
        folds[cat == 5] = 2.0
        if self.float16_correction:
            # TorNet's dealiased velocities were computed in float16 (vel + k*nyq with
            # float16 nyquist), which shifts unfolded values by up to ~0.05 m/s.
            corr = (folds * 2.0).astype(np.float16) * np.float16(nyquist)
            out = (arrays.astype(np.float16) + corr).astype(np.float32)
            out = np.where(folds == 0, arrays, out)
        else:
            out = arrays + folds * np.float32(2.0 * nyquist)
        if n_az == 2 * N_RADIALS_1DEG:
            out = _merge_super_res(out)
        else:
            out = out[0]
        return out.astype(np.float32)
