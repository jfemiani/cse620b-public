import * as pdfjs from "./vendor/pdfjs/pdf.mjs";

pdfjs.GlobalWorkerOptions.workerSrc = new URL("./vendor/pdfjs/pdf.worker.mjs", import.meta.url).href;
const status = document.querySelector("#presentation-status");
const title = document.querySelector("#presentation-title");
const download = document.querySelector("#pdf-download");
const slides = document.querySelector(".slides");
const rendered = new Map();
let pdf;
let deck;
let renderSequence = 0;

function reportError(error) {
  status.hidden = false;
  status.setAttribute("role", "alert");
  status.textContent = `Unable to display this presentation: ${error.message}. Use the PDF download if available.`;
}

async function renderSlide(index) {
  if (rendered.has(index)) return rendered.get(index);
  const task = (async () => {
    const section = slides.children[index];
    section.textContent = "Loading slide...";
    const page = await pdf.getPage(index + 1);
    const viewport = page.getViewport({ scale: 1.5 });
    const canvas = document.createElement("canvas");
    canvas.width = Math.ceil(viewport.width);
    canvas.height = Math.ceil(viewport.height);
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", `Slide ${index + 1}. Download the PDF for selectable text.`);
    await page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;
    section.replaceChildren(canvas);
    page.cleanup();
  })();
  rendered.set(index, task);
  try {
    await task;
  } catch (error) {
    rendered.delete(index);
    throw error;
  }
}

async function showCurrentSlide() {
  const sequence = ++renderSequence;
  const index = deck.getIndices().h;
  await renderSlide(index);
  if (sequence !== renderSequence) return;
  if (index + 1 < pdf.numPages) await renderSlide(index + 1);
  if (sequence !== renderSequence) return;
  for (const [position, task] of rendered) {
    if (Math.abs(position - index) > 2) {
      await task;
      if (sequence !== renderSequence) return;
      slides.children[position].replaceChildren();
      rendered.delete(position);
    }
  }
}

try {
  const params = new URLSearchParams(location.search);
  const path = params.get("pdf");
  if (!path) throw new Error("No PDF was selected");
  const pdfUrl = new URL(path, location.href);
  const lessonRoot = new URL("Lessons/", location.href);
  if (pdfUrl.origin !== location.origin ||
      !pdfUrl.pathname.startsWith(lessonRoot.pathname) ||
      !pdfUrl.pathname.toLowerCase().endsWith(".pdf")) {
    throw new Error("Select a PDF from the course Lessons directory");
  }
  title.textContent = params.get("title") || "Lecture slideshow";
  document.title = `${title.textContent} | CSE 620B`;
  download.href = pdfUrl.href;
  download.hidden = false;
  pdf = await pdfjs.getDocument({
    url: pdfUrl.href,
    standardFontDataUrl: new URL("./vendor/standard_fonts/", import.meta.url).href,
    cMapUrl: new URL("./vendor/cmaps/", import.meta.url).href,
    cMapPacked: true,
    isEvalSupported: false,
  }).promise;
  const firstPage = await pdf.getPage(1);
  const size = firstPage.getViewport({ scale: 1 });
  for (let index = 0; index < pdf.numPages; index++) {
    const section = document.createElement("section");
    section.setAttribute("aria-label", `Slide ${index + 1} of ${pdf.numPages}`);
    slides.append(section);
  }
  deck = new window.Reveal(document.querySelector(".reveal"), {
    width: size.width,
    height: size.height,
    margin: 0.04,
    hash: true,
    slideNumber: "c/t",
    transition: "none",
    controls: true,
    embedded: true,
  });
  await deck.initialize();
  deck.on("slidechanged", () => { showCurrentSlide().catch(reportError); });
  await showCurrentSlide();
  status.hidden = true;
} catch (error) {
  reportError(error);
}
