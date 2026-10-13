import { LocalAIError } from "../errors";
function softmax(values) {
    let max = -Infinity;
    for (let i = 0; i < values.length; i++)
        max = Math.max(max, values[i]);
    const exponentials = Array.from(values, (value) => Math.exp(value - max));
    const total = exponentials.reduce((sum, value) => sum + value, 0);
    return exponentials.map((value) => value / total);
}
export function processOutput(outputs, manifest, labels) {
    const first = Object.values(outputs)[0];
    if (!first)
        throw new LocalAIError("INFERENCE_FAILED", "Model returned no outputs.");
    if (manifest.output.processor === "softmax") {
        const scores = softmax(first.data);
        if (labels && labels.length !== scores.length) {
            throw new LocalAIError("INFERENCE_FAILED", "Label count does not match model output size.");
        }
        const results = scores.map((score, index) => ({
            label: labels?.[index] ?? String(index),
            score,
            index,
        }));
        const topK = typeof manifest.output.topK === "number" ? manifest.output.topK : 5;
        return results
            .sort((left, right) => right.score - left.score)
            .slice(0, Math.max(1, Math.floor(topK)));
    }
    return outputs;
}
//# sourceMappingURL=output.js.map