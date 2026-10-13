export type LocalAIErrorCode =
  | "MANIFEST_LOAD_FAILED"
  | "INVALID_MANIFEST"
  | "MODEL_DOWNLOAD_FAILED"
  | "CACHE_FAILED"
  | "WEBGPU_UNAVAILABLE"
  | "RUNTIME_INIT_FAILED"
  | "INVALID_INPUT"
  | "INFERENCE_FAILED";

export class LocalAIError extends Error {
  readonly code: LocalAIErrorCode;

  constructor(code: LocalAIErrorCode, message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "LocalAIError";
    this.code = code;
  }
}
