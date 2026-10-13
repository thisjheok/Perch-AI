import * as ort from "onnxruntime-web";
import { LocalAIError } from "../errors";
import { isWebGPUAvailable } from "./capability";
export async function createSession(model, runtime) {
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
        }
        catch {
            // WebGPU may be exposed but unusable. Continue with the portable runtime.
        }
    }
    return {
        session: await createWithProvider(model, "wasm"),
        runtime: "wasm",
    };
}
async function createWithProvider(model, provider) {
    try {
        return await ort.InferenceSession.create(model, {
            executionProviders: [provider],
        });
    }
    catch (cause) {
        throw new LocalAIError("RUNTIME_INIT_FAILED", `Could not initialize the ${provider} runtime.`, { cause });
    }
}
//# sourceMappingURL=session.js.map