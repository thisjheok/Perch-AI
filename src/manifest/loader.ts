import { LocalAIError } from "../errors";
import type { ResolvedModelManifest } from "./schema";
import { validateManifest } from "./validator";

export async function loadManifest(manifestUrl: string): Promise<ResolvedModelManifest> {
  let response: Response;
  try {
    response = await fetch(manifestUrl);
  } catch (cause) {
    throw new LocalAIError("MANIFEST_LOAD_FAILED", `Could not load manifest: ${manifestUrl}.`, { cause });
  }

  if (!response.ok) {
    throw new LocalAIError(
      "MANIFEST_LOAD_FAILED",
      `Could not load manifest: ${manifestUrl} (${response.status}).`,
    );
  }

  let raw: unknown;
  try {
    raw = await response.json();
  } catch (cause) {
    throw new LocalAIError("MANIFEST_LOAD_FAILED", "Manifest is not valid JSON.", { cause });
  }

  const manifest = validateManifest(raw);
  const baseUrl = new URL(manifestUrl, globalThis.location?.href ?? "http://localhost/");
  const assetUrls = Object.fromEntries(
    Object.entries(manifest.files).map(([name, path]) => [name, new URL(path, baseUrl).href]),
  );

  return {
    ...manifest,
    manifestUrl: baseUrl.href,
    modelUrl: assetUrls.model,
    assetUrls,
  };
}
