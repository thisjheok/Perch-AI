export const MANIFEST_SCHEMA_VERSION = 1 as const;

export const SUPPORTED_TASKS = [
  "text-classification",
  "embedding",
  "image-classification",
  "recommendation",
  "regression",
] as const;

export type ModelTask = (typeof SUPPORTED_TASKS)[number];

export interface ModelManifest {
  schemaVersion: typeof MANIFEST_SCHEMA_VERSION;
  id: string;
  version: string;
  task: ModelTask;
  runtime: "onnx";
  files: {
    model: string;
    [asset: string]: string;
  };
  sha256?: Record<string, string>;
  input: {
    type: string;
    processor?: string;
    maxLength?: number;
    [property: string]: unknown;
  };
  output: {
    type: string;
    processor?: string;
    [property: string]: unknown;
  };
}

export interface ResolvedModelManifest extends ModelManifest {
  manifestUrl: string;
  modelUrl: string;
  assetUrls: Record<string, string>;
}
