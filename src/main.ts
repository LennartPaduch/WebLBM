import { CanvasPainter } from "./canvas";
import { GPUController } from "./GPUController";
import { LBM, VisColormaps, VisTypes } from "./LBM";

try {
  const gpu = await GPUController.create();

  const Nx = 1 << 10;
  const Ny = 1 << 10;
  const lbm = new LBM(Nx, Ny, gpu);
  await lbm.init();
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  const painter = new CanvasPainter({
    canvas,
    Nx,
    Ny,
    onPaint: (rows, value) => lbm.applyMaskRows(rows, value),
  });
  painter.enable();
  lbm.run();

  const restartBtn = document.getElementById("restart-btn");
  restartBtn?.addEventListener("click", () => lbm.run());

  const resetCanvasBtn = document.getElementById("reset-btn");
  resetCanvasBtn?.addEventListener("click", () => lbm.resetMask());

  const colormapSelect = document.getElementById(
    "colormap-select"
  ) as HTMLSelectElement | null;
  if (colormapSelect) {
    colormapSelect.innerHTML = "";
    colormapSelect.className =
      "w-full px-3 py-2 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 cursor-pointer hover:bg-gray-650 transition-colors";

    for (const [label, value] of Object.entries(VisColormaps)) {
      const opt = document.createElement("option");
      opt.value = String(value);
      opt.textContent = label;
      opt.className = "bg-gray-700 text-white";
      colormapSelect.appendChild(opt);
    }

    colormapSelect.addEventListener("change", () => {
      lbm.setVisColormap(Number(colormapSelect.value) as any);
    });
  }

  const visTypeSelect = document.getElementById(
    "visType-select"
  ) as HTMLSelectElement | null;
  if (visTypeSelect) {
    visTypeSelect.innerHTML = "";
    visTypeSelect.className =
      "w-full px-3 py-2 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 cursor-pointer hover:bg-gray-650 transition-colors";

    for (const [label, value] of Object.entries(VisTypes)) {
      const opt = document.createElement("option");
      opt.value = String(value);
      opt.textContent = label;
      opt.className = "bg-gray-700 text-white";
      visTypeSelect.appendChild(opt);
    }

    visTypeSelect.addEventListener("change", () => {
      lbm.setVisType(Number(visTypeSelect.value) as any);
    });
  }

  // Paint Options
  const paintSettings: HTMLDivElement = document.getElementById(
    "paint-settings"
  ) as HTMLDivElement;

  // Brush Mode Buttons
  const brushModeBtnWrapper = document.createElement("div");
  brushModeBtnWrapper.className = "flex gap-2";

  const eraseModeBtn = document.createElement("button");
  eraseModeBtn.textContent = "Erase";
  eraseModeBtn.className =
    "flex-1 px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-all border border-gray-600";
  eraseModeBtn.ariaLabel = "Erase obstacles";

  const paintModeBtn = document.createElement("button");
  paintModeBtn.textContent = "Paint";
  paintModeBtn.className =
    "flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-500 transition-all border-2 border-blue-400 shadow-lg shadow-blue-500/30";
  paintModeBtn.ariaLabel = "Paint obstacles";

  paintModeBtn.addEventListener("click", () => {
    paintModeBtn.className =
      "flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-500 transition-all border-2 border-blue-400 shadow-lg shadow-blue-500/30";
    eraseModeBtn.className =
      "flex-1 px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-all border border-gray-600";
    painter.setModeSolid();
  });

  eraseModeBtn.addEventListener("click", () => {
    eraseModeBtn.className =
      "flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-500 transition-all border-2 border-blue-400 shadow-lg shadow-blue-500/30";
    paintModeBtn.className =
      "flex-1 px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-all border border-gray-600";
    painter.setModeErase();
  });

  brushModeBtnWrapper.appendChild(paintModeBtn);
  brushModeBtnWrapper.appendChild(eraseModeBtn);
  paintSettings.appendChild(brushModeBtnWrapper);

  // Brush Size Slider
  const brushSizeSliderWrapper = document.createElement("div");
  brushSizeSliderWrapper.className = "flex items-center gap-3";

  const brushSizeSliderLabel = document.createElement("label");
  brushSizeSliderLabel.textContent = "Size:";
  brushSizeSliderLabel.htmlFor = "brush-size";
  brushSizeSliderLabel.className =
    "text-sm font-medium text-gray-300 min-w-fit";

  const brushSizeSlider = document.createElement("input");
  brushSizeSlider.type = "range";
  brushSizeSlider.min = "1";
  brushSizeSlider.max = "100";
  brushSizeSlider.value = "6";
  brushSizeSlider.id = "brush-size";
  brushSizeSlider.className =
    "flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500";

  const brushSizeValue = document.createElement("span");
  brushSizeValue.textContent = brushSizeSlider.value;
  brushSizeValue.className =
    "text-sm font-medium text-blue-400 min-w-[2rem] text-right";

  brushSizeSlider.addEventListener("input", (e) => {
    const value = Number((e.target as HTMLInputElement).value);
    brushSizeValue.textContent = String(value);
    painter.setBrush(value);
  });

  brushSizeSliderWrapper.appendChild(brushSizeSliderLabel);
  brushSizeSliderWrapper.appendChild(brushSizeSlider);
  brushSizeSliderWrapper.appendChild(brushSizeValue);
  paintSettings.appendChild(brushSizeSliderWrapper);

  // \Paint Options

  function resizeCanvasSquare() {
    const cssSide = Math.min(window.innerWidth, window.innerHeight);
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = Math.floor(cssSide * dpr);
    canvas.height = Math.floor(cssSide * dpr);
  }

  window.addEventListener("resize", resizeCanvasSquare);
  window.addEventListener("orientationchange", resizeCanvasSquare);
  resizeCanvasSquare();
} catch (e) {
  showError(e);
}

function showError(message: unknown) {
  document.getElementById("wrapper")?.classList.add("hidden");

  const root = document.createElement("div");
  root.className = "fixed inset-0 z-50 grid place-items-center p-4";

  const card = document.createElement("div");
  card.setAttribute("role", "alert");
  card.className =
    "w-full max-w-md rounded-2xl border border-zinc-200 bg-white shadow-xl " +
    "dark:border-zinc-800 dark:bg-zinc-900";

  const safe =
    typeof message === "string"
      ? message
      : message instanceof Error
      ? message.message
      : "Something went wrong.";

  card.innerHTML = `
    <div class="p-4 sm:p-5">
      <div class="flex items-start gap-3">
        <span class="mt-0.5 inline-flex h-6 w-6 items-center justify-center rounded-full
          bg-red-100 text-red-600 dark:bg-red-900/30">!</span>
        <p class="text-sm text-zinc-700 dark:text-zinc-200 break-words"></p>
      </div>
    </div>
  `;

  (card.querySelector("p") as HTMLParagraphElement).textContent = String(safe);

  root.appendChild(card);
  document.body.appendChild(root);
}
