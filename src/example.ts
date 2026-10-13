import "./example.css";
import { pipeline } from "@huggingface/transformers";
import { createLocalAI, type LocalAI } from "./index";
import { imageToTensor } from "./example-image";

const manifestUrl = document.querySelector<HTMLInputElement>("#manifest-url")!;
const fileInput = document.querySelector<HTMLInputElement>("#image-file")!;
const dropZone = document.querySelector<HTMLDivElement>("#drop-zone")!;
const preview = document.querySelector<HTMLImageElement>("#preview")!;
const loadButton = document.querySelector<HTMLButtonElement>("#load")!;
const runButton = document.querySelector<HTMLButtonElement>("#run")!;
const status = document.querySelector<HTMLParagraphElement>("#status")!;
const output = document.querySelector<HTMLPreElement>("#output")!;
const loadLmButton = document.querySelector<HTMLButtonElement>("#load-lm")!;
const askLmButton = document.querySelector<HTMLButtonElement>("#ask-lm")!;
const lmStatus = document.querySelector<HTMLParagraphElement>("#lm-status")!;
const promptInput = document.querySelector<HTMLTextAreaElement>("#prompt")!;
const lmOutput = document.querySelector<HTMLPreElement>("#lm-output")!;

let ai: LocalAI | undefined;
let selectedFile: File | undefined;
let previewUrl: string | undefined;
type TextGenerator = (text: string, options: Record<string, unknown>) => Promise<unknown>;
let textGenerator: TextGenerator | undefined;

function setStatus(message: string): void {
  status.textContent = message;
}

function selectImage(file: File): void {
  if (!file.type.startsWith("image/")) {
    setStatus("이미지 파일만 선택할 수 있습니다.");
    return;
  }
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.hidden = false;
  runButton.disabled = !ai;
  setStatus(`${file.name} 선택됨`);
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) selectImage(file);
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("is-dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("is-dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("is-dragging");
  const file = event.dataTransfer?.files[0];
  if (file) selectImage(file);
});

loadButton.addEventListener("click", async () => {
  loadButton.disabled = true;
  runButton.disabled = true;
  try {
    ai = await createLocalAI({
      model: { manifestUrl: manifestUrl.value },
      runtime: "auto",
      onProgress: ({ loaded, total }) => {
        setStatus(`모델 다운로드 중: ${loaded} / ${total ?? "?"} bytes`);
      },
    });
    setStatus(`모델 준비 완료 (${ai.runtime})`);
    runButton.disabled = !selectedFile;
  } catch (error) {
    setStatus(error instanceof Error ? error.message : "모델 로드에 실패했습니다.");
  } finally {
    loadButton.disabled = false;
  }
});

runButton.addEventListener("click", async () => {
  if (!ai || !selectedFile) return;
  runButton.disabled = true;
  try {
    const tensor = await imageToTensor(selectedFile);
    const result = await ai.run({ input: tensor });
    output.textContent = JSON.stringify(result, null, 2);
  } catch (error) {
    output.textContent = error instanceof Error ? error.message : "추론에 실패했습니다.";
  } finally {
    runButton.disabled = false;
  }
});

loadLmButton.addEventListener("click", async () => {
  loadLmButton.disabled = true;
  try {
    textGenerator = await pipeline("text2text-generation", "Xenova/flan-t5-small", {
      dtype: "q8",
      device: "wasm",
      progress_callback: (progress: { status?: string; progress?: number }) => {
        if (progress.status === "progress") {
          lmStatus.textContent = `LM 다운로드 중: ${Math.round(progress.progress ?? 0)}%`;
        }
      },
    }) as unknown as TextGenerator;
    lmStatus.textContent = "LM 준비 완료";
    askLmButton.disabled = false;
  } catch (error) {
    lmStatus.textContent = error instanceof Error ? error.message : "LM 로드에 실패했습니다.";
  } finally {
    loadLmButton.disabled = false;
  }
});

askLmButton.addEventListener("click", async () => {
  if (!textGenerator || !promptInput.value.trim()) return;
  askLmButton.disabled = true;
  try {
    const result = await textGenerator(promptInput.value, {
      max_new_tokens: 60,
      do_sample: false,
    });
    const generated = Array.isArray(result) && result[0] && "generated_text" in result[0]
      ? String(result[0].generated_text)
      : "";
    lmOutput.textContent = generated || JSON.stringify(result, null, 2) || "(생성 결과가 비어 있습니다.)";
  } catch (error) {
    lmOutput.textContent = error instanceof Error ? error.message : "생성에 실패했습니다.";
  } finally {
    askLmButton.disabled = false;
  }
});
