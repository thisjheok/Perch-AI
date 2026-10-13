import * as ort from "onnxruntime-web";
export type Runtime = "auto" | "webgpu" | "wasm";
export type SelectedRuntime = Exclude<Runtime, "auto">;
export declare function createSession(model: ArrayBuffer, runtime: Runtime): Promise<{
    session: ort.InferenceSession;
    runtime: SelectedRuntime;
}>;
//# sourceMappingURL=session.d.ts.map