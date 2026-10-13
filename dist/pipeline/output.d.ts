import type { ModelManifest } from "../manifest/schema";
import type { ModelOutputs } from "../index";
export interface ClassificationResult {
    label: string;
    score: number;
    index: number;
}
export declare function processOutput(outputs: ModelOutputs, manifest: ModelManifest, labels?: string[]): unknown;
//# sourceMappingURL=output.d.ts.map