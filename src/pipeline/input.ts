import { LocalAIError } from "../errors";
import type { ModelManifest } from "../manifest/schema";
import type { ModelInputs } from "../index";

export function prepareInput(
  input: unknown,
  manifest: ModelManifest,
): ModelInputs {
  if (manifest.input.type !== "tensor") {
    throw new LocalAIError(
      "INVALID_INPUT",
      `Input type '${manifest.input.type}' is not supported yet. Use type 'tensor'.`,
    );
  }
  if (typeof input !== "object" || input === null || Array.isArray(input)) {
    throw new LocalAIError("INVALID_INPUT", "Model input must be a tensor map.");
  }
  return input as ModelInputs;
}
