import * as ort from "onnxruntime-web";

export async function imageToTensor(file: File): Promise<ort.Tensor> {
  const bitmap = await createImageBitmap(file);
  const canvas = document.createElement("canvas");
  canvas.width = 224;
  canvas.height = 224;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas 2D context is unavailable.");

  const scale = Math.max(224 / bitmap.width, 224 / bitmap.height);
  const width = bitmap.width * scale;
  const height = bitmap.height * scale;
  context.drawImage(bitmap, (224 - width) / 2, (224 - height) / 2, width, height);
  bitmap.close();

  const pixels = context.getImageData(0, 0, 224, 224).data;
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
  return new ort.Tensor("float32", data, [1, 3, 224, 224]);
}
