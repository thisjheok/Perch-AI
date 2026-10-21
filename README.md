<p align="center">
  <img src="icon.png" alt="Perch icon" width="160">
</p>

[English](README.md) | [한국어](README.ko.md)

# Perch Browser ONNX SDK

**Perch** is a browser-side SDK for loading and running lightweight ONNX models. It uses ONNX Runtime Web, preferring WebGPU when available and supporting WASM for broader compatibility. Model files are loaded separately from your application bundle using a JSON manifest.

> This repository is an early-stage project (`0.1.0`). Check the API and runtime behavior against your target models and browsers before relying on it in production.

## Features

- Load a model and its assets from a manifest URL.
- Select `auto`, `webgpu`, or `wasm` runtime.
- Report model download progress.
- Use browser Cache API-backed model downloads.
- Run ONNX tensors and optionally process classification output with softmax and labels.

## Requirements

- Node.js and npm
- A modern browser
- A model exported to ONNX with a manifest describing its files and input/output behavior

## Install and run the example

```sh
npm install
npm run dev
```

Open the local URL printed by Vite. The example page demonstrates image classification with the bundled MobileNetV2 model and also includes a separate text-generation demo powered by Transformers.js. The text-generation path is an example-page feature, not part of the `createLocalAI()` API.

Open `/benchmark.html` to profile preprocessing across multiple images and compare available FP32, FP16, and INT8 MobileNetV2 variants across WASM/WebGPU. Preprocessing reports p50/p95 separately for decode, canvas resize, pixel readback, and tensor packing. Model comparisons report model size, cold/warm initialization, p50/p95 inference latency, and output differences from FP32; model tests run 30 inferences per trial and three trials by default. INT8 requires representative calibration images.

Generate model variants with the Python tool. Static INT8 quantization requires at least 10 representative input images (labels are not required). Do not calibrate with a logo or other non-representative images; the resulting quality would not predict real inputs. The generator also reports top-1/top-5 agreement and probability differences against FP32 on the calibration images; because calibration and comparison use the same images, this is not independent validation.

```sh
python -m pip install -r tools/requirements-quantization.txt
python tools/quantize_mobilenet.py --calibration-dir path/to/representative-images
```

FP16 needs no calibration data. INT8 statically quantizes Conv/Gemm with the supplied images and validates the ONNX and CPU inference. Generated variants are written under `public/models/mobilenetv2/`; the benchmark discovers their manifests automatically. The generator leaves the FP32 source model unchanged.

## SDK usage

```ts
import { createLocalAI } from "@perch-ai/browser";
import { Tensor } from "onnxruntime-web";

const ai = await createLocalAI({
  model: { manifestUrl: "/models/mobilenetv2/manifest.json" },
  runtime: "auto",
  onProgress: ({ loaded, total, phase }) => {
    console.log(`${phase}: ${loaded} / ${total ?? "unknown"} bytes`);
  },
});

try {
  const input = new Tensor("float32", inputData, inputDimensions);
  const result = await ai.run({ input });
  console.log(result);
} finally {
  await ai.dispose();
}
```

Supply the tensor name and dimensions expected by your ONNX model; preprocessing (such as resizing and normalization) is the application’s responsibility. The bundled image example contains a preprocessing implementation.

`createLocalAI()` returns `modelId`, the selected `runtime`, `run(input)`, and `dispose()`. Runtime choices are `auto` (tries WebGPU, then WASM during session creation), `webgpu`, and `wasm`. Explicit WebGPU requires browser support. The current automatic fallback happens during session initialization, not after an inference failure.

## Model manifest

A manifest describes the ONNX model and optional assets. For example, `public/models/mobilenetv2/manifest.json`:

```json
{
  "schemaVersion": 1,
  "id": "mobilenetv2-imagenet",
  "version": "12.0.0",
  "task": "image-classification",
  "runtime": "onnx",
  "files": {
    "model": "model.onnx",
    "labels": "labels.json"
  },
  "input": { "type": "tensor" },
  "output": {
    "type": "labels",
    "processor": "softmax",
    "topK": 5
  }
}
```

File paths are resolved relative to the manifest URL. The example assets are in `public/models/`; Vite serves them under `/models/` during development. Input tensors must match the model’s expected ONNX inputs. With the `softmax` output processor, the SDK maps scores to the optional string-array labels asset and returns the configured top results.

## Development commands

```sh
npm run dev        # Start the Vite development server
npm run build      # Type-check and build declarations/JavaScript
npm test           # Run tests once
npm run test:watch # Run tests in watch mode
```

## Project layout

- `src/index.ts` — public SDK API
- `src/manifest/` — manifest loading and validation
- `src/model/` — model download and cache handling
- `src/runtime/` — ONNX Runtime Web session selection
- `src/pipeline/` — input checks and output processing
- `src/example.ts` — interactive browser example
- `src/benchmark.ts` — browser WASM/WebGPU benchmark
- `public/models/` — example manifests, model files, and labels
- `docs/` — design and implementation notes

## Current scope

This project runs pre-converted ONNX models; it does not train or convert models. The SDK expects callers to provide appropriately named ONNX tensors and does not currently perform image preprocessing or validate tensor dimensions against the model. Browser runtime availability and model compatibility vary. Review the design and learning notes in `docs/` for implementation details and limitations.

## License

MIT (see the `license` field in `package.json`).
