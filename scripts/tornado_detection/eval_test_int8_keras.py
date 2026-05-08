"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import keras
import numpy as np
import tqdm

from tornet.data import preprocess as pp
from tornet.data.constants import ALL_VARIABLES
from tornet.data.loader import read_file
from tornet.metrics.keras import metrics as tfm
from tornet.models.keras.layers import CoordConv2D, FillNaNs


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = "/mnt/low-priority/edgewarn/TorNet"
DEFAULT_REPRESENTATIVE_BATCHES = 32
DEFAULT_BATCH_SIZE = 64
NETCDF_SUFFIXES = {".nc", ".nc4", ".netcdf"}
POSITIVE_PREFIX = "TOR"
NEGATIVE_PREFIXES = {"WRN", "NUL"}


class FileListKerasDataLoader(keras.utils.PyDataset):
    """Minimal Keras-compatible loader for an explicit list of test files."""

    def __init__(
        self,
        file_list,
        batch_size=DEFAULT_BATCH_SIZE,
        include_az=False,
        select_keys=None,
        tilt_last=True,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=10,
    ):
        super().__init__(workers, use_multiprocessing, max_queue_size)
        self.file_list = list(file_list)
        self.batch_size = batch_size
        self.include_az = include_az
        self.select_keys = select_keys
        self.tilt_last = tilt_last

    def __len__(self):
        return int(np.ceil(len(self.file_list) / self.batch_size))

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.file_list))
        files_batch = self.file_list[low:high]

        element_list = []
        for file_path in files_batch:
            element = read_file(file_path, variables=ALL_VARIABLES, n_frames=1, tilt_last=self.tilt_last)
            element["label"] = np.array([infer_label_from_filename(file_path)], dtype=element["label"].dtype)
            pp.add_coordinates(element, include_az=self.include_az, backend=np, tilt_last=self.tilt_last)
            element["coordinates"] = element["coordinates"][None, ...]
            element_list.append(element)

        batch = {}
        for key in element_list[0].keys():
            batch[key] = np.concatenate([element[key] for element in element_list])

        x, y = pp.split_x_y(batch)
        return pp.select_keys(x, keys=self.select_keys), y


def discover_test_files(data_root):
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    file_list = []
    for year_dir in sorted(path for path in data_root.iterdir() if path.is_dir() and path.name.isdigit()):
        test_dir = year_dir / "test"
        if not test_dir.is_dir():
            continue

        for nested_year_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
            for candidate in nested_year_dir.rglob("*"):
                if candidate.is_file() and candidate.suffix.lower() in NETCDF_SUFFIXES:
                    file_list.append(str(candidate))

    if not file_list:
        raise RuntimeError(
            "No test files found. Expected files under directories shaped like YYYY/test/YYYY/*.nc"
        )

    return sorted(file_list)


def infer_label_from_filename(file_path):
    prefix = Path(file_path).name.split("_", 1)[0].upper()
    if prefix == POSITIVE_PREFIX:
        return 1
    if prefix in NEGATIVE_PREFIXES:
        return 0
    raise RuntimeError(
        f"Unable to infer binary label from filename prefix for {file_path}. "
        "Expected prefixes: TOR, WRN, or NUL."
    )


def count_labels(file_list):
    counts = {"TOR": 0, "WRN": 0, "NUL": 0}
    for file_path in file_list:
        prefix = Path(file_path).name.split("_", 1)[0].upper()
        if prefix not in counts:
            raise RuntimeError(
                f"Unable to infer binary label from filename prefix for {file_path}. "
                "Expected prefixes: TOR, WRN, or NUL."
            )
        counts[prefix] += 1
    return counts


def get_input_keys(model):
    return [tensor.name.split(":")[0] for tensor in model.inputs]


def normalize_tensor_name(name):
    return name.split(":", 1)[0].split("/", 1)[0].lower()


def get_tflite_input_pairs(input_details, input_keys):
    key_map = {normalize_tensor_name(key): key for key in input_keys}
    pairs = []
    for detail in input_details:
        detail_name = normalize_tensor_name(detail["name"])
        if detail_name not in key_map:
            raise RuntimeError(
                f"Unable to match TFLite input {detail['name']} to model inputs {input_keys}"
            )
        pairs.append((detail, key_map[detail_name]))
    return pairs


def make_metrics():
    from_logits = True
    return [
        keras.metrics.AUC(from_logits=from_logits, name="AUC", num_thresholds=2000),
        keras.metrics.AUC(from_logits=from_logits, curve="PR", name="AUCPR", num_thresholds=2000),
        tfm.BinaryAccuracy(from_logits=from_logits, name="BinaryAccuracy"),
        tfm.Precision(from_logits=from_logits, name="Precision"),
        tfm.Recall(from_logits=from_logits, name="Recall"),
        tfm.F1Score(from_logits=from_logits, name="F1"),
    ]


def metric_results(metrics):
    return {metric.name: float(metric.result().numpy()) for metric in metrics}


def representative_dataset(dataset, input_keys, max_batches):
    batch_total = min(len(dataset), max_batches)
    for batch_index in tqdm.tqdm(
        range(batch_total),
        desc="INT8 calibration",
        unit="batch",
    ):
        x_batch, _ = dataset[batch_index]
        yield [x_batch[key].astype(np.float32) for key in input_keys]


def quantize_array(array, scale, zero_point, dtype):
    if scale == 0:
        return array.astype(dtype)
    if np.issubdtype(dtype, np.integer):
        q = np.round(array / scale + zero_point)
        q = np.clip(q, np.iinfo(dtype).min, np.iinfo(dtype).max)
        return q.astype(dtype)
    return array.astype(dtype)


def dequantize_array(array, scale, zero_point):
    if scale == 0:
        return array.astype(np.float32)
    return (array.astype(np.float32) - zero_point) * scale


def evaluate_keras_model(model, dataset):
    metrics = make_metrics()
    batch_count = len(dataset)
    sample_count = 0

    start_time = time.perf_counter()
    for batch_index in tqdm.tqdm(range(batch_count), desc="Float32 eval", unit="batch"):
        x_batch, y_batch = dataset[batch_index]
        logits = model(x_batch, training=False)
        for metric in metrics:
            metric.update_state(y_batch, logits)
        sample_count += int(y_batch.shape[0])
    elapsed = time.perf_counter() - start_time

    results = metric_results(metrics)
    results.update(
        {
            "num_batches": batch_count,
            "num_samples": sample_count,
            "total_seconds": elapsed,
            "seconds_per_sample": elapsed / max(sample_count, 1),
        }
    )
    return results


def convert_to_int8_tflite(model, dataset, input_keys, output_path, representative_batches):
    import tensorflow as tf

    specs = []
    for tensor in model.inputs:
        shape = [None if dim is None else int(dim) for dim in tensor.shape]
        specs.append(
            tf.TensorSpec(
                shape=shape,
                dtype=tensor.dtype,
                name=tensor.name.split(":")[0],
            )
        )

    @tf.function(input_signature=specs)
    def serving_fn(*args):
        x_batch = {key: value for key, value in zip(input_keys, args)}
        return model(x_batch, training=False)

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [serving_fn.get_concrete_function()],
        model,
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(
        dataset, input_keys, representative_batches
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    with open(output_path, "wb") as handle:
        handle.write(tflite_model)


def evaluate_tflite_model(tflite_path, dataset, input_keys):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_pairs = get_tflite_input_pairs(input_details, input_keys)

    metrics = make_metrics()
    batch_count = len(dataset)
    sample_count = 0

    start_time = time.perf_counter()
    for batch_index in tqdm.tqdm(range(batch_count), desc="INT8 eval", unit="batch"):
        x_batch, y_batch = dataset[batch_index]

        needs_resize = any(
            int(detail["shape"][0]) != int(x_batch[key].shape[0])
            for detail, key in input_pairs
        )
        if needs_resize:
            for detail, key in input_pairs:
                interpreter.resize_tensor_input(detail["index"], x_batch[key].shape, strict=False)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            input_pairs = get_tflite_input_pairs(input_details, input_keys)

        for detail, key in input_pairs:
            scale, zero_point = detail["quantization"]
            value = quantize_array(x_batch[key].astype(np.float32), scale, zero_point, detail["dtype"])
            interpreter.set_tensor(detail["index"], value)

        interpreter.invoke()

        output_detail = output_details[0]
        output_tensor = interpreter.get_tensor(output_detail["index"])
        out_scale, out_zero_point = output_detail["quantization"]
        logits = dequantize_array(output_tensor, out_scale, out_zero_point)

        for metric in metrics:
            metric.update_state(y_batch, logits)
        sample_count += int(y_batch.shape[0])
    elapsed = time.perf_counter() - start_time

    results = metric_results(metrics)
    results.update(
        {
            "num_batches": batch_count,
            "num_samples": sample_count,
            "total_seconds": elapsed,
            "seconds_per_sample": elapsed / max(sample_count, 1),
        }
    )
    return results


def load_base_model_path():
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id="tornet-ml/tornado_detector_baseline_v1",
        filename="tornado_detector_baseline.keras",
    )


def load_base_model(model_path):
    return keras.saving.load_model(
        model_path,
        compile=False,
        custom_objects={
            "CoordConv2D": CoordConv2D,
            "FillNaNs": FillNaNs,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT, help="Root of the nested test dataset")
    parser.add_argument(
        "--int8_model_path",
        default="./tornado_detector_baseline_int8.tflite",
        help="Where to save the generated INT8 TFLite model",
    )
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--representative_batches",
        type=int,
        default=DEFAULT_REPRESENTATIVE_BATCHES,
        help="Number of test batches to use for INT8 calibration",
    )
    parser.add_argument("--output_json", default=None, help="Optional JSON file for results")
    args = parser.parse_args()

    if keras.config.backend() != "tensorflow":
        raise RuntimeError("INT8 TFLite evaluation requires KERAS_BACKEND=tensorflow")

    model_path = load_base_model_path()
    model = load_base_model(model_path)
    input_keys = get_input_keys(model)

    file_list = discover_test_files(args.data_root)
    label_counts = count_labels(file_list)
    dataset = FileListKerasDataLoader(
        file_list=file_list,
        batch_size=args.batch_size,
        include_az=False,
        select_keys=input_keys,
        workers=1,
        use_multiprocessing=False,
    )

    LOGGER.info("Using backend %s", keras.config.backend())
    LOGGER.info("Loaded %d test files from %s", len(file_list), args.data_root)
    LOGGER.info("Label counts: %s", label_counts)
    LOGGER.info("Model inputs: %s", input_keys)

    float_results = evaluate_keras_model(model, dataset)

    convert_to_int8_tflite(
        model=model,
        dataset=dataset,
        input_keys=input_keys,
        output_path=args.int8_model_path,
        representative_batches=args.representative_batches,
    )
    int8_results = evaluate_tflite_model(args.int8_model_path, dataset, input_keys)

    results = {
        "data_root": os.path.abspath(args.data_root),
        "model_path": os.path.abspath(model_path),
        "int8_model_path": os.path.abspath(args.int8_model_path),
        "num_test_files": len(file_list),
        "label_counts": label_counts,
        "batch_size": args.batch_size,
        "representative_batches": min(args.representative_batches, len(dataset)),
        "float32": float_results,
        "int8": int8_results,
    }

    print(json.dumps(results, indent=2, sort_keys=True))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
