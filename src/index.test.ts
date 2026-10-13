import { beforeEach, describe, expect, it, vi } from "vitest";
import { createLocalAI } from "./index";

const { create } = vi.hoisted(() => ({
  create: vi.fn(),
}));

vi.mock("onnxruntime-web", () => ({
  InferenceSession: { create },
}));

describe("createLocalAI", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.unstubAllGlobals();
    vi.stubGlobal("fetch", vi.fn());
  });

  it("loads a manifest, resolves the model URL, and runs a WASM session", async () => {
    const outputs = { output: { data: [1], dims: [1] } };
    const run = vi.fn().mockResolvedValue(outputs);
    const release = vi.fn().mockResolvedValue(undefined);
    create.mockResolvedValue({ run, release });
    vi.mocked(fetch)
      .mockResolvedValueOnce(
      new Response(JSON.stringify({
        schemaVersion: 1,
        id: "test-model",
        version: "1.0.0",
        task: "text-classification",
        runtime: "onnx",
        files: { model: "model.onnx" },
        input: { type: "tensor" },
        output: { type: "labels" },
      }), { status: 200 }),
      )
      .mockResolvedValueOnce(new Response(new Uint8Array([1, 2, 3]), { status: 200 }));

    const ai = await createLocalAI({
      model: { manifestUrl: "https://cdn.example.com/models/manifest.json" },
    });
    const input = { input: { data: [1], dims: [1] } } as never;

    await expect(ai.run(input)).resolves.toBe(outputs);
    await ai.dispose();

    const [modelBuffer, sessionOptions] = create.mock.calls[0] as [ArrayBuffer, unknown];
    expect(Array.from(new Uint8Array(modelBuffer))).toEqual([1, 2, 3]);
    expect(sessionOptions).toEqual({ executionProviders: ["wasm"] });
    expect(run).toHaveBeenCalledWith(input);
    expect(release).toHaveBeenCalledOnce();
    expect(ai.modelId).toBe("test-model");
  });

  it("returns an error code for an invalid manifest", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify({ schemaVersion: 99 }), { status: 200 }),
    );

    const error = await createLocalAI({
      model: { manifestUrl: "https://cdn.example.com/manifest.json" },
    }).catch((value: unknown) => value as { code: string }) as unknown as { code: string };

    expect(error.code).toBe("INVALID_MANIFEST");
    expect(create).not.toHaveBeenCalled();
  });

  it("returns an error code when the manifest cannot be loaded", async () => {
    vi.mocked(fetch).mockResolvedValue(new Response("not found", { status: 404 }));

    const error = await createLocalAI({
      model: { manifestUrl: "https://cdn.example.com/manifest.json" },
    }).catch((value: unknown) => value as { code: string }) as unknown as { code: string };

    expect(error.code).toBe("MANIFEST_LOAD_FAILED");
  });

  it("falls back to WASM when WebGPU session initialization fails", async () => {
    const release = vi.fn().mockResolvedValue(undefined);
    create
      .mockRejectedValueOnce(new Error("unsupported WebGPU operator"))
      .mockResolvedValueOnce({ run: vi.fn(), release });
    vi.stubGlobal("navigator", { gpu: {} });
    vi.mocked(fetch)
      .mockResolvedValueOnce(new Response(JSON.stringify({
        schemaVersion: 1,
        id: "fallback-model",
        version: "1.0.0",
        task: "embedding",
        runtime: "onnx",
        files: { model: "model.onnx" },
        input: { type: "tensor" },
        output: { type: "embedding" },
      }), { status: 200 }))
      .mockResolvedValueOnce(new Response(new Uint8Array([1]), { status: 200 }));

    const ai = await createLocalAI({
      model: { manifestUrl: "https://cdn.example.com/manifest.json" },
    });

    expect(ai.runtime).toBe("wasm");
    expect(create.mock.calls.map((call) => call[1])).toEqual([
      { executionProviders: ["webgpu"] },
      { executionProviders: ["wasm"] },
    ]);
  });

  it("rejects explicit WebGPU when it is unavailable", async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(new Response(JSON.stringify({
        schemaVersion: 1,
        id: "gpu-model",
        version: "1.0.0",
        task: "embedding",
        runtime: "onnx",
        files: { model: "model.onnx" },
        input: { type: "tensor" },
        output: { type: "embedding" },
      }), { status: 200 }))
      .mockResolvedValueOnce(new Response(new Uint8Array([1]), { status: 200 }));

    const error = await createLocalAI({
      model: { manifestUrl: "https://cdn.example.com/manifest.json" },
      runtime: "webgpu",
    }).catch((value: unknown) => value as { code: string }) as unknown as { code: string };

    expect(error.code).toBe("WEBGPU_UNAVAILABLE");
  });
});
