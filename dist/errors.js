export class LocalAIError extends Error {
    code;
    constructor(code, message, options) {
        super(message, options);
        this.name = "LocalAIError";
        this.code = code;
    }
}
//# sourceMappingURL=errors.js.map