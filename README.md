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

Open `/benchmark.html` to compare WASM and WebGPU with the same image tensor and bundled MobileNetV2 model. It reports preprocessing, cold-cache initialization (including model download), warm-cache initialization, and p50/p95 inference latency after warm-up separately. Before a cold initialization it clears only this model's SDK Cache API entries; the browser HTTP cache is not controlled. Runtime order alternates across trials, with 30 inferences per trial and three trials by default. The JSON report includes the environment and actual execution order. It checks top-1 agreement and top-K score differences, not ground-truth accuracy.

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
