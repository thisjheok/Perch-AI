import * as ort from "onnxruntime-web";

export interface ImagePreprocessingTimings {
  decodeMs: number;
  resizeMs: number;
  readbackMs: number;
  tensorPackMs: number;
  totalMs: number;
}

export async function imageToTensor(
  file: File,
  onTimings?: (timings: ImagePreprocessingTimings) => void,
): Promise<ort.Tensor> {
  const totalStarted = performance.now();
  const bitmap = await createImageBitmap(file);
  const decodeMs = performance.now() - totalStarted;
  const resizeStarted = performance.now();
  const canvas = document.createElement("canvas");
  canvas.width = 224;
  canvas.height = 224;
  const context = canvas.getContext("2d");
  if (!context) {
    bitmap.close();
    throw new Error("Canvas 2D context is unavailable.");
  }

  const scale = Math.max(224 / bitmap.width, 224 / bitmap.height);
  const width = bitmap.width * scale;
  const height = bitmap.height * scale;
  context.drawImage(bitmap, (224 - width) / 2, (224 - height) / 2, width, height);
  bitmap.close();
  const resizeMs = performance.now() - resizeStarted;

  const readbackStarted = performance.now();
  const pixels = context.getImageData(0, 0, 224, 224).data;
  const readbackMs = performance.now() - readbackStarted;
  const tensorPackStarted = performance.now();
  const data = new Float32Array(1 * 3 * 224 * 224);
  for (let y = 0; y < 224; y++) {
    for (let x = 0; x < 224; x++) {
      const pixel = (y * 224 + x) * 4;
      const position = y * 224 + x;
      data[position] = (pixels[pixel] / 255 - 0.485) / 0.229;
      data[224 * 224 + position] = (pixels[pixel + 1] / 255 - 0.456) / 0.224;
      data[2 * 224 * 224 + position] = (pixels[pixel + 2] / 255 - 0.406) / 0.225;
    }
  }
  const tensor = new ort.Tensor("float32", data, [1, 3, 224, 224]);
  const tensorPackMs = performance.now() - tensorPackStarted;
  onTimings?.({
    decodeMs,
    resizeMs,
    readbackMs,
    tensorPackMs,
    totalMs: performance.now() - totalStarted,
  });
  return tensor;
}
