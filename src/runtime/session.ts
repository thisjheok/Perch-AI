import * as ort from "onnxruntime-web";
import { LocalAIError } from "../errors";
import { isWebGPUAvailable } from "./capability";

export type Runtime = "auto" | "webgpu" | "wasm";
export type SelectedRuntime = Exclude<Runtime, "auto">;

export async function createSession(
  model: ArrayBuffer,
  runtime: Runtime,
): Promise<{ session: ort.InferenceSession; runtime: SelectedRuntime }> {
  if (runtime === "wasm") {
    return {
      session: await createWithProvider(model, "wasm"),
      runtime: "wasm",
    };
  }

  if (runtime === "webgpu") {
    if (!isWebGPUAvailable()) {
      throw new LocalAIError("WEBGPU_UNAVAILABLE", "WebGPU is not available.");
    }
    return {
      session: await createWithProvider(model, "webgpu"),
      runtime: "webgpu",
    };
  }

  if (isWebGPUAvailable()) {
    try {
      return {
        session: await createWithProvider(model, "webgpu"),
        runtime: "webgpu",
      };
    } catch {
      // WebGPU may be exposed but unusable. Continue with the portable runtime.
    }
  }

  return {
    session: await createWithProvider(model, "wasm"),
    runtime: "wasm",
  };
}

async function createWithProvider(
  model: ArrayBuffer,
  provider: SelectedRuntime,
): Promise<ort.InferenceSession> {
  try {
    return await ort.InferenceSession.create(model, {
      executionProviders: [provider],
    });
  } catch (cause) {
    throw new LocalAIError(
      "RUNTIME_INIT_FAILED",
      `Could not initialize the ${provider} runtime.`,
      { cause },
    );
  }
}
