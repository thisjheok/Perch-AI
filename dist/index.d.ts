import * as ort from "onnxruntime-web";
import { type DownloadProgress } from "./model/downloader";
import { type Runtime, type SelectedRuntime } from "./runtime/session";
export type { LocalAIErrorCode } from "./errors";
export interface CreateLocalAIOptions {
    model: {
        manifestUrl: string;
    };
    runtime?: Runtime;
    onProgress?: (progress: DownloadProgress & {
        phase: "model";
    }) => void;
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
export declare function createLocalAI(options: CreateLocalAIOptions): Promise<LocalAI>;
//# sourceMappingURL=index.d.ts.map