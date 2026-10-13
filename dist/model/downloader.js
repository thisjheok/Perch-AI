import { LocalAIError } from "../errors";
const CACHE_NAME = "perch-ai-models-v1";
function cacheKey(url) {
    return `https://perch-ai.invalid/model?url=${encodeURIComponent(url)}`;
}
async function readResponse(response, onProgress) {
    const totalHeader = response.headers.get("content-length");
    const total = totalHeader ? Number(totalHeader) : undefined;
    if (!response.body) {
        const buffer = await response.arrayBuffer();
        onProgress?.({ loaded: buffer.byteLength, total });
        return buffer;
    }
    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;
    while (true) {
        const { done, value } = await reader.read();
        if (done)
            break;
        if (value) {
            chunks.push(value);
            loaded += value.byteLength;
            onProgress?.({ loaded, total });
        }
    }
    const buffer = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
        buffer.set(chunk, offset);
        offset += chunk.byteLength;
    }
    return buffer.buffer;
}
export async function downloadModel(url, onProgress) {
    try {
        if (typeof caches !== "undefined") {
            const cache = await caches.open(CACHE_NAME);
            const cached = await cache.match(cacheKey(url));
            if (cached) {
                const buffer = await cached.arrayBuffer();
                onProgress?.({ loaded: buffer.byteLength, total: buffer.byteLength });
                return buffer;
            }
        }
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const buffer = await readResponse(response, onProgress);
        if (typeof caches !== "undefined") {
            try {
                const cache = await caches.open(CACHE_NAME);
                await cache.put(cacheKey(url), new Response(buffer));
            }
            catch (cause) {
                throw new LocalAIError("CACHE_FAILED", `Could not cache model: ${url}.`, { cause });
            }
        }
        return buffer;
    }
    catch (cause) {
        if (cause instanceof LocalAIError)
            throw cause;
        throw new LocalAIError("MODEL_DOWNLOAD_FAILED", `Could not download model: ${url}.`, { cause });
    }
}
//# sourceMappingURL=downloader.js.map