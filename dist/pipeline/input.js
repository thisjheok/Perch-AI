import { LocalAIError } from "../errors";
export function prepareInput(input, manifest) {
    if (manifest.input.type !== "tensor") {
        throw new LocalAIError("INVALID_INPUT", `Input type '${manifest.input.type}' is not supported yet. Use type 'tensor'.`);
    }
    if (typeof input !== "object" || input === null || Array.isArray(input)) {
        throw new LocalAIError("INVALID_INPUT", "Model input must be a tensor map.");
    }
    return input;
}
//# sourceMappingURL=input.js.map