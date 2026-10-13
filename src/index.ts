import * as ort from "onnxruntime-web";
import { LocalAIError } from "./errors";
import { loadManifest } from "./manifest/loader";
import { downloadModel, type DownloadProgress } from "./model/downloader";
import { createSession, type Runtime, type SelectedRuntime } from "./runtime/session";
import { prepareInput } from "./pipeline/input";
import { processOutput } from "./pipeline/output";

export type { LocalAIErrorCode } from "./errors";

export interface CreateLocalAIOptions {
  model: {
    manifestUrl: string;
  };
  runtime?: Runtime;
  onProgress?: (progress: DownloadProgress & { phase: "model" }) => void;
}

export type TensorInput = ort.Tensor;

export type ModelInputs = Record<string, TensorInput>;
export type ModelOutputs = Record<string, ort.Tensor>;

export interface LocalAI {
  readonly modelId: string;
  readonly runtime: SelectedRuntime;
  run(input: unknown): Promise<unknown>;
  dispose(): Promise<void>;
}

export async function createLocalAI(
  options: CreateLocalAIOptions,
): Promise<LocalAI> {
  if (!options.model.manifestUrl) {
    throw new LocalAIError("MANIFEST_LOAD_FAILED", "A manifest URL is required.");
  }

  const manifest = await loadManifest(options.model.manifestUrl);
  const model = await downloadModel(manifest.modelUrl, (progress) => {
    options.onProgress?.({ ...progress, phase: "model" });
  });
  let labels: string[] | undefined;
  if (manifest.assetUrls.labels) {
    const labelsBuffer = await downloadModel(manifest.assetUrls.labels);
    try {
      const parsed: unknown = JSON.parse(new TextDecoder().decode(labelsBuffer));
      if (!Array.isArray(parsed) || !parsed.every((label) => typeof label === "string")) {
        throw new Error("labels must be an array of strings");
      }
      labels = parsed;
    } catch (cause) {
      throw new LocalAIError("INVALID_MANIFEST", "The labels asset is invalid JSON.", { cause });
    }
  }
  const selected = await createSession(model, options.runtime ?? "auto");

  return {
    modelId: manifest.id,
    runtime: selected.runtime,
    async run(input) {
      const outputs = await selected.session.run(prepareInput(input, manifest));
      return processOutput(outputs, manifest, labels);
    },
    async dispose() {
      await selected.session.release();
    },
  };
}
