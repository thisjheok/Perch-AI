import {
  MANIFEST_SCHEMA_VERSION,
  SUPPORTED_TASKS,
  type ModelManifest,
} from "./schema";
import { LocalAIError } from "../errors";

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requireString(
  value: Record<string, unknown>,
  key: string,
  path: string,
): string {
  if (typeof value[key] !== "string" || value[key].trim() === "") {
    throw new LocalAIError("INVALID_MANIFEST", `${path}.${key} must be a non-empty string.`);
  }
  return value[key];
}

export function validateManifest(value: unknown): ModelManifest {
  if (!isObject(value)) {
    throw new LocalAIError("INVALID_MANIFEST", "Manifest must be a JSON object.");
  }

  if (value.schemaVersion !== MANIFEST_SCHEMA_VERSION) {
    throw new LocalAIError(
      "INVALID_MANIFEST",
      `Unsupported manifest schemaVersion: ${String(value.schemaVersion)}.`,
    );
  }

  const id = requireString(value, "id", "manifest");
  const version = requireString(value, "version", "manifest");
  const runtime = requireString(value, "runtime", "manifest");
  const task = requireString(value, "task", "manifest");

  if (runtime !== "onnx") {
    throw new LocalAIError("INVALID_MANIFEST", `Unsupported runtime: ${runtime}.`);
  }
  if (!(SUPPORTED_TASKS as readonly string[]).includes(task)) {
    throw new LocalAIError("INVALID_MANIFEST", `Unsupported task: ${task}.`);
  }

  if (!isObject(value.files)) {
    throw new LocalAIError("INVALID_MANIFEST", "manifest.files must be an object.");
  }
  const files = { ...value.files };
  const model = requireString(files, "model", "manifest.files");

  if (!isObject(value.input)) {
    throw new LocalAIError("INVALID_MANIFEST", "manifest.input must be an object.");
  }
  if (!isObject(value.output)) {
    throw new LocalAIError("INVALID_MANIFEST", "manifest.output must be an object.");
  }
  const input = { ...value.input, type: requireString(value.input, "type", "manifest.input") };
  const output = { ...value.output, type: requireString(value.output, "type", "manifest.output") };

  return {
    schemaVersion: MANIFEST_SCHEMA_VERSION,
    id,
    version,
    task: task as ModelManifest["task"],
    runtime: "onnx",
    files: { ...files, model },
    sha256: isObject(value.sha256) ? { ...value.sha256 } as Record<string, string> : undefined,
    input,
    output,
  };
}
