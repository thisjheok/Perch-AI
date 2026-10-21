from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image, ImageOps
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_static,
)
from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.transformers.onnx_model import OnnxModel


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "public" / "models" / "mobilenetv2"
SOURCE_MODEL = MODEL_DIR / "model.onnx"
SOURCE_MANIFEST = MODEL_DIR / "manifest.json"
CALIBRATION_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


class ImageCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_paths: list[Path], input_name: str) -> None:
        self.input_name = input_name
        self.image_paths = image_paths
        self.rewind()

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            image_path = next(self.image_iterator)
        except StopIteration:
            return None
        with Image.open(image_path) as image:
            image = ImageOps.fit(image.convert("RGB"), (224, 224), method=Image.Resampling.BILINEAR)
            pixels = np.asarray(image, dtype=np.float32) / 255.0
        tensor = np.stack(
            (
                (pixels[:, :, 0] - 0.485) / 0.229,
                (pixels[:, :, 1] - 0.456) / 0.224,
                (pixels[:, :, 2] - 0.406) / 0.225,
            ),
            axis=0,
        )[None, ...]
        return {self.input_name: tensor}

    def rewind(self) -> None:
        self.image_iterator = iter(self.image_paths)


def validate_and_smoke_test(model_path: Path) -> None:
    onnx.checker.check_model(str(model_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    input_data = np.zeros((1, 3, 224, 224), dtype=np.float32)
    outputs = session.run(None, {model_input.name: input_data})
    if not outputs or any(not np.isfinite(output).all() for output in outputs):
        raise ValueError(f"Model produced missing or non-finite outputs: {model_path}")


def softmax(logits: np.ndarray) -> np.ndarray:
    values = logits.astype(np.float64)
    values -= values.max(axis=-1, keepdims=True)
    exponentials = np.exp(values)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def compare_calibration_outputs(model_path: Path, image_paths: list[Path], input_name: str) -> None:
    reference = ort.InferenceSession(str(SOURCE_MODEL), providers=["CPUExecutionProvider"])
    candidate = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    reader = ImageCalibrationDataReader(image_paths, input_name)
    top1_matches = 0
    top5_matches = 0
    max_probability_differences: list[float] = []

    for _ in image_paths:
        sample = reader.get_next()
        if sample is None:
            raise ValueError("Calibration image reader ended earlier than expected.")
        reference_logits = reference.run(None, sample)[0]
        candidate_logits = candidate.run(None, {candidate.get_inputs()[0].name: sample[input_name]})[0]
        reference_scores = softmax(reference_logits)[0]
        candidate_scores = softmax(candidate_logits)[0]
        reference_top5 = np.argsort(reference_scores)[-5:][::-1]
        candidate_top5 = np.argsort(candidate_scores)[-5:][::-1]
        top1_matches += int(reference_top5[0] == candidate_top5[0])
        top5_matches += int(set(reference_top5) == set(candidate_top5))
        max_probability_differences.append(float(np.max(np.abs(reference_scores - candidate_scores))))

    print(
        f"{model_path.stem} calibration-set parity (not independent validation): "
        f"top-1 {top1_matches}/{len(image_paths)}, top-5 {top5_matches}/{len(image_paths)}, "
        f"mean/worst max probability difference "
        f"{np.mean(max_probability_differences):.6f}/{max(max_probability_differences):.6f}"
    )


def write_manifest(base: dict[str, object], *, variant: str, model_file: str) -> Path:
    manifest = dict(base)
    manifest["id"] = f"mobilenetv2-imagenet-{variant}"
    manifest["version"] = f"{base['version']}-{variant}"
    files = dict(base["files"])
    files["model"] = model_file
    manifest["files"] = files

    path = MODEL_DIR / f"manifest-{variant}.json"
    path.write_text(f"{json.dumps(manifest, indent=2)}\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate FP16 MobileNetV2 and optional calibrated INT8 ONNX variants."
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        help="Directory with at least 10 representative RGB images for static INT8 quantization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))

    fp16_path = MODEL_DIR / "model-fp16.onnx"
    fp16_model = convert_float_to_float16(onnx.load(SOURCE_MODEL), keep_io_types=True)
    fp16_wrapper = OnnxModel(fp16_model)
    fp16_wrapper.topological_sort()
    fp16_model = fp16_wrapper.model
    onnx.checker.check_model(fp16_model)
    onnx.save_model(fp16_model, fp16_path)
    validate_and_smoke_test(fp16_path)
    fp16_manifest = write_manifest(base_manifest, variant="fp16", model_file=fp16_path.name)

    original_size = SOURCE_MODEL.stat().st_size
    print(f"FP32: {original_size:,} bytes, manifest: {SOURCE_MANIFEST.relative_to(ROOT)}")
    print(
        f"FP16: {fp16_path.stat().st_size:,} bytes "
        f"({fp16_path.stat().st_size / original_size:.1%} of FP32), "
        f"manifest: {fp16_manifest.relative_to(ROOT)}"
    )

    if args.calibration_dir is None:
        print("INT8 skipped: provide --calibration-dir with representative images.")
        return

    calibration_dir = args.calibration_dir.resolve()
    image_paths = sorted(
        path for path in calibration_dir.iterdir()
        if path.is_file() and path.suffix.lower() in CALIBRATION_EXTENSIONS
    )
    if len(image_paths) < 10:
        raise ValueError(
            f"INT8 calibration needs at least 10 images; found {len(image_paths)} in {calibration_dir}."
        )

    int8_path = MODEL_DIR / "model-int8.onnx"
    with TemporaryDirectory(prefix="perch-onnx-preprocess-") as temporary_directory:
        preprocessed_path = Path(temporary_directory) / "mobilenetv2-preprocessed.onnx"
        quant_pre_process(SOURCE_MODEL, preprocessed_path)
        preprocessed_model = onnx.load(preprocessed_path)
        quantize_static(
            preprocessed_path,
            int8_path,
            ImageCalibrationDataReader(image_paths, preprocessed_model.graph.input[0].name),
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=["Conv", "Gemm"],
            per_channel=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )
    validate_and_smoke_test(int8_path)
    int8_manifest = write_manifest(base_manifest, variant="int8", model_file=int8_path.name)
    print(
        f"INT8: {int8_path.stat().st_size:,} bytes "
        f"({int8_path.stat().st_size / original_size:.1%} of FP32), "
        f"calibrated with {len(image_paths)} images, "
        f"manifest: {int8_manifest.relative_to(ROOT)}"
    )
    input_name = onnx.load(SOURCE_MODEL).graph.input[0].name
    compare_calibration_outputs(fp16_path, image_paths, input_name)
    compare_calibration_outputs(int8_path, image_paths, input_name)


if __name__ == "__main__":
    main()
