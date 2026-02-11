import { VisColormaps, VisTypes, type VisType, type VisColormap } from "./LBM";
import { makeBuildContext, Presets, resolveParams } from "./presets";
import { SimulationController } from "./SimulationController";


export class UIController {
  // Private UI Element References
  #settingsPanel!: HTMLDivElement;
  #showSettingsBtn!: HTMLButtonElement;
  #hideSettingsBtn!: HTMLButtonElement;
  #canvas!: HTMLCanvasElement;
  #resSelect!: HTMLSelectElement;
  #colormapSelect!: HTMLSelectElement;
  #visTypeSelect!: HTMLSelectElement;
  #sceneSelect!: HTMLSelectElement;
  #togglePauseBtn!: HTMLButtonElement;
  #resetPaintBtn!: HTMLButtonElement;
  #brushSizeValue!: HTMLSpanElement;
  #mlupsDisplay!: HTMLSpanElement;

  // Powers of two keep wrap indexing cheap in shaders (bitmasking instead of modulo).
  #PRESETS = [
    1 << 8,/*256*/
    1 << 9, /*512*/
    1 << 10, /*1024*/
    1 << 11, /*2048*/
  ];

  #controller: SimulationController;
  #maxDim: number;
  #maxCap: number;

  #isBusy = false;
  #isSettingsVisible = true;
  #smallViewportQuery: MediaQueryList;

  #lastPollTime = 0;
  #lastStepCount = 0;
  #pollIntervalId: number = 0;

  constructor(
    controller: SimulationController,
    maxDim: number,
    maxCap: number,
  ) {
    this.#controller = controller;
    this.#maxDim = maxDim;
    this.#maxCap = maxCap;
    this.#smallViewportQuery = window.matchMedia("(max-width: 900px)");

    this.#initElements();
    this.#setupListeners();
  }

  /**
   * Identifies existing DOM elements and dynamically builds the rest.
   */
  #initElements() {
    this.#settingsPanel = document.getElementById("settings") as HTMLDivElement;
    this.#showSettingsBtn = document.getElementById(
      "show-settings-btn",
    ) as HTMLButtonElement;
    this.#hideSettingsBtn = document.getElementById(
      "hide-settings-btn",
    ) as HTMLButtonElement;
    this.#canvas = document.getElementById("canvas") as HTMLCanvasElement;

    const paintSettings = document.getElementById(
      "paint-settings",
    ) as HTMLDivElement;
    this.#sceneSelect = document.getElementById(
      "mask-scene-select",
    ) as HTMLSelectElement;
    this.#colormapSelect = document.getElementById(
      "colormap-select",
    ) as HTMLSelectElement;
    this.#visTypeSelect = document.getElementById(
      "visType-select",
    ) as HTMLSelectElement;
    this.#togglePauseBtn = document.getElementById(
      "togglePause-btn",
    ) as HTMLButtonElement;

    // 1. Build Resolution Select
    this.#resSelect = this.#createStyledSelect("resolution-select");
    this.#refreshResolutionOptions();

    const resWrapper = this.#createLabeledWrapper(
      "Resolution (N×N):",
      this.#resSelect,
    );
    paintSettings.prepend(resWrapper);

    // 2. Populate Colormap & VisType
    this.#populateSelect(this.#colormapSelect, VisColormaps);
    this.#populateSelect(this.#visTypeSelect, VisTypes);

    // 3. Populate Presets
    Object.keys(Presets).forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      this.#sceneSelect.appendChild(opt);
    });

    // 4. Build Brush Controls
    this.#createBrushControls(paintSettings);

    const statusContainer = document.createElement("div");
    statusContainer.className =
      "mt-4 px-1 text-xs font-mono text-gray-400";
    this.#mlupsDisplay = document.createElement("span");
    this.#mlupsDisplay.className = "block whitespace-pre-line";
    this.#mlupsDisplay.textContent = "Total: Waiting...\nFluid: Waiting...";

    const metricsLabel = document.createElement("span");
    metricsLabel.className = "block";
    metricsLabel.textContent = "End-to-End MLUPS:";
    metricsLabel.title =
      "Computed as (cells * deltaSteps) / deltaTime using wall-clock time. Includes simulation, visualization, and rendering/frame pacing. Total counts all cells; Fluid excludes solid-mask cells.";

    statusContainer.appendChild(metricsLabel);
    statusContainer.appendChild(this.#mlupsDisplay);

    document.getElementById("paint-settings")?.appendChild(statusContainer);

    // Mobile starts with pure rendering view; desktop starts with settings visible.
    this.#setSettingsVisible(!this.#isSmallViewport());
  }

  #startMetricsLoop() {
    if (this.#pollIntervalId) {
      clearInterval(this.#pollIntervalId);
    }

    // We use setInterval because we don't need frame-perfect syncing for text updates.
    // 500ms is a good balance between readability and responsiveness.
    this.#pollIntervalId = window.setInterval(() => {
      // If we are recreating the GPU buffers, do not attempt to read from LBM
      if (this.#isBusy || !this.#controller.lbm) {
        this.#mlupsDisplay.className = "block whitespace-pre-line";
        this.#mlupsDisplay.textContent = "Total: ---\nFluid: ---";
        return;
      }

      const now = performance.now();
      const currentSteps = this.#controller.lbm.tick;
      const totalCellCount = this.#controller.lbm.getCellcount();
      const fluidCellCount = this.#controller.lbm.getFluidCellcount();

      // Avoid divide by zero on first run
      if (this.#lastPollTime > 0) {
        const dt = now - this.#lastPollTime; // ms
        const dSteps = currentSteps - this.#lastStepCount;

        // MLUPS = (Cells * Steps) / (Seconds * 1,000,000)
        // (Cells * Steps) / (ms * 1000)
        if (dt > 0 && dSteps >= 0) {
          const totalMlups = (totalCellCount * dSteps) / (dt * 1000);
          const fluidMlups = (fluidCellCount * dSteps) / (dt * 1000);
          this.#mlupsDisplay.textContent =
            `Total: ${totalMlups.toFixed(2)} MLUPS\nFluid: ${fluidMlups.toFixed(2)} MLUPS`;

          // Optional: Color code performance
          this.#mlupsDisplay.className = "block whitespace-pre-line text-green-400";
        } else if (dSteps < 0) {
          // Counter reset (e.g. restart): start a fresh measurement window.
          this.#lastPollTime = now;
          this.#lastStepCount = currentSteps;
          return;
        }
      }

      this.#lastPollTime = now;
      this.#lastStepCount = currentSteps;
    }, 500);
  }

  // This ensures inputs are visually disabled AND logically ignored
  async #withBusyLock(fn: () => Promise<void>) {
    if (this.#isBusy) return;

    try {
      this.#setBusy(true);
      await fn();

      // Reset metrics counters so we don't get a huge spike after a long load time
      if (this.#controller.lbm) {
        this.#lastPollTime = performance.now();
        this.#lastStepCount = this.#controller.lbm.tick;
      }
    } catch (e) {
      console.error("Critical UI Operation Failed", e);
    } finally {
      this.#setBusy(false);
    }
  }

  #setBusy(busy: boolean) {
    this.#isBusy = busy;
    const opacity = busy ? "0.5" : "1.0";
    const pointerEvents = busy ? "none" : "auto";

    // List of heavy inputs
    const controls = [
      this.#resSelect,
      this.#sceneSelect,
      this.#colormapSelect,
      this.#visTypeSelect,
      this.#togglePauseBtn,
      this.#resetPaintBtn,
      this.#showSettingsBtn,
      this.#hideSettingsBtn,
    ];

    controls.forEach((el) => {
      el.style.opacity = opacity;
      el.style.pointerEvents = pointerEvents;
      if (el instanceof HTMLButtonElement || el instanceof HTMLSelectElement) {
        el.disabled = busy;
      }
    });

    // Also disable canvas painting interaction if desired
    // this.#controller.painter.canvas.style.pointerEvents = pointerEvents;
  }

  /**
   * Attaches all event logic.
   */
  #setupListeners() {
    // 1. Start the Polling Loop (Independent of Physics Loop)
    this.#startMetricsLoop();

    this.#bindFastPress(this.#showSettingsBtn, () => this.#setSettingsVisible(true));
    this.#bindFastPress(this.#hideSettingsBtn, () => this.#setSettingsVisible(false));
    const onViewportChange = () => this.#syncViewportLayout();
    if (typeof this.#smallViewportQuery.addEventListener === "function") {
      this.#smallViewportQuery.addEventListener("change", onViewportChange);
    } else {
      this.#smallViewportQuery.addListener(onViewportChange);
    }

    // 2. Wrap heavy listeners in the Lock
    this.#sceneSelect.addEventListener("change", () => {
      this.#withBusyLock(async () => {
        const presetName = this.#sceneSelect.value;
        const preset = Presets[presetName];
        if (!preset) return;
        this.#controller.currentPresetName = presetName;

        // Recreate if the preset changes resolution.
        if (preset.resolution && preset.resolution !== this.#controller.N) {
          await this.#controller.recreate(
            preset.resolution,
            this.#maxDim,
            this.#maxCap,
            presetName,
          );
          this.syncAll();
          return;
        }

        const result = this.#buildPresetResult(presetName);
        if (!result) return;

        // Apply mask
        this.#controller.lbm.setMask(result);

        // Apply visualization preferences after simulation parameters are updated.
        if (result.visType !== undefined) this.#updateVisType(result.visType);
        if (result.colormap !== undefined) this.#updateColormap(result.colormap);

        this.#controller.lbm.restart();
        this.#syncPauseButton();
      });
    });

    this.#resSelect.addEventListener("change", () => {
      this.#withBusyLock(async () => {
        await this.#controller.recreate(
          Number(this.#resSelect.value),
          this.#maxDim,
          this.#maxCap,
          this.#sceneSelect.value,
        );
        this.syncAll();
      });
    });

    // Vis & Colormap Changes
    this.#colormapSelect.addEventListener("change", () =>
      this.#updateColormap(Number(this.#colormapSelect.value)),
    );
    this.#visTypeSelect.addEventListener("change", () =>
      this.#updateVisType(Number(this.#visTypeSelect.value)),
    );

    // Controls
    this.#togglePauseBtn.addEventListener("click", () => {
      this.#controller.lbm.togglePause();
      this.#syncPauseButton();
    });

    document
      .getElementById("restart-btn")
      ?.addEventListener("click", () => {
        this.#controller.lbm.restart();
        this.#syncPauseButton();
        this.#lastPollTime = performance.now();
        this.#lastStepCount = this.#controller.lbm.tick;
      });
  }

  syncAll() {
    this.#resSelect.value = String(this.#controller.N);
    if (this.#controller.currentPresetName && Presets[this.#controller.currentPresetName]) {
      this.#sceneSelect.value = this.#controller.currentPresetName;
    }
    this.#syncPauseButton();
    const preset = Presets[this.#sceneSelect.value];
    if (preset) {
      if (preset.visType !== undefined)
        this.#visTypeSelect.value = String(preset.visType);
      if (preset.colormap !== undefined)
        this.#colormapSelect.value = String(preset.colormap);
    }
  }

  // --- Helper Update Methods ---

  #updateColormap(val: number) {
    this.#colormapSelect.value = String(val);
    this.#controller.lbm.setVisColormap(val as VisColormap);
  }

  #updateVisType(val: number) {
    this.#visTypeSelect.value = String(val);
    if (val === VisTypes.VORTICITY) {
      // On mode switch, default to a diverging map suited for signed vorticity.
      this.#updateColormap(VisColormaps.RdBu);
    }
    this.#controller.lbm.setVisType(val as VisType);
  }

  #syncPauseButton() {
    this.#togglePauseBtn.textContent = this.#controller.lbm.isRunning()
      ? "Pause"
      : "Resume";
  }

  #isSmallViewport() {
    return this.#smallViewportQuery.matches;
  }

  #setSettingsVisible(visible: boolean) {
    this.#isSettingsVisible = visible;
    this.#settingsPanel.classList.toggle("hidden", !visible);
    this.#showSettingsBtn.classList.toggle("hidden", visible);
    this.#syncViewportLayout();
  }

  #syncViewportLayout() {
    if (!this.#isSmallViewport()) {
      this.#canvas.classList.remove("opacity-0", "pointer-events-none");
      return;
    }
    this.#canvas.classList.toggle("opacity-0", this.#isSettingsVisible);
    this.#canvas.classList.toggle("pointer-events-none", this.#isSettingsVisible);
  }

  #bindFastPress(button: HTMLButtonElement, action: () => void) {
    let swallowClick = false;
    button.addEventListener("pointerdown", (e) => {
      if (e.pointerType !== "touch" && e.pointerType !== "pen") return;
      swallowClick = true;
      e.preventDefault();
      action();
    });
    button.addEventListener("click", () => {
      if (swallowClick) {
        swallowClick = false;
        return;
      }
      action();
    });
  }

  // --- DOM Construction Helpers ---

  #refreshResolutionOptions() {
    this.#resSelect.innerHTML = "";
    const allowed = this.#PRESETS.filter(
      (n) => n <= this.#maxDim && n <= this.#maxCap,
    );
    allowed.forEach((size) => {
      const opt = document.createElement("option");
      opt.value = String(size);
      opt.textContent = `${size} × ${size}`;
      if (size === this.#controller.N) opt.selected = true;
      this.#resSelect.appendChild(opt);
    });
  }

  #createBrushControls(parent: HTMLElement) {
    const wrapper = document.createElement("div");
    wrapper.className = "grid grid-cols-2 gap-2";

    const paintBtn = this.#createButton(
      "Paint",
      "bg-blue-600 border-blue-400 shadow-blue-500/30",
    );
    const eraseBtn = this.#createButton("Erase", "bg-gray-700");
    this.#resetPaintBtn = this.#createButton(
      "Reset Paint",
      "bg-gray-700 hover:bg-gray-600",
    );

    paintBtn.onclick = () => {
      this.#controller.painter.setModeSolid();
      paintBtn.classList.add("border-2", "shadow-lg");
      eraseBtn.classList.remove("border-2", "shadow-lg");
    };

    eraseBtn.onclick = () => {
      this.#controller.painter.setModeErase();
      eraseBtn.classList.add("border-2", "shadow-lg");
      paintBtn.classList.remove("border-2", "shadow-lg");
    };

    this.#resetPaintBtn.onclick = () => {
      this.#withBusyLock(async () => {
        if (!this.#controller.lbm) return;
        const presetName = this.#sceneSelect.value;
        const result = this.#buildPresetResult(presetName);
        if (!result) return;
        this.#controller.currentPresetName = presetName;
        this.#controller.lbm.setMask(result);
        this.#controller.lbm.restart();
        this.#syncPauseButton();
      });
    };

    wrapper.append(paintBtn, eraseBtn);
    parent.appendChild(wrapper);

    const resetWrapper = document.createElement("div");
    resetWrapper.className = "flex gap-2 mt-2";
    resetWrapper.append(this.#resetPaintBtn);
    parent.appendChild(resetWrapper);

    const sliderWrapper = document.createElement("div");
    sliderWrapper.className = "flex items-center gap-3 mt-2";

    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "1";
    slider.max = "100";
    slider.value = "3";
    slider.className =
      "flex-1 h-2 bg-gray-700 rounded-lg appearance-none accent-blue-500";

    this.#brushSizeValue = document.createElement("span");
    this.#brushSizeValue.className =
      "text-sm text-blue-400 min-w-[2rem] text-right";
    this.#brushSizeValue.textContent = "3";

    slider.oninput = () => {
      this.#brushSizeValue.textContent = slider.value;
      this.#controller.painter.setBrush(Number(slider.value));
    };

    sliderWrapper.append(slider, this.#brushSizeValue);
    parent.appendChild(sliderWrapper);
  }

  #populateSelect(select: HTMLSelectElement, data: Record<string, any>) {
    select.innerHTML = "";
    for (const [label, value] of Object.entries(data)) {
      const opt = document.createElement("option");
      opt.value = String(value);
      opt.textContent = label;
      select.appendChild(opt);
    }
  }

  #buildPresetResult(presetName: string) {
    const preset = Presets[presetName];
    if (!preset) return null;
    const params = resolveParams(preset /*, uiParamOverrides */);
    const ctx = makeBuildContext(this.#controller.N, this.#controller.N, presetName, params);
    return preset.build(ctx);
  }

  #createStyledSelect(id: string): HTMLSelectElement {
    const s = document.createElement("select");
    s.id = id;
    s.className =
      "w-full px-3 py-2 bg-gray-700 text-white rounded-lg border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500/50 cursor-pointer hover:bg-gray-600 transition-colors";
    return s;
  }

  #createLabeledWrapper(labelText: string, element: HTMLElement): HTMLElement {
    const div = document.createElement("div");
    div.className = "flex flex-col gap-1 sm:flex-row sm:items-center sm:gap-3";
    const label = document.createElement("label");
    label.textContent = labelText;
    label.className = "text-sm font-medium text-gray-300 sm:min-w-fit";
    div.append(label, element);
    return div;
  }

  #createButton(text: string, extraClasses: string): HTMLButtonElement {
    const btn = document.createElement("button");
    btn.textContent = text;
    btn.className = `w-full px-4 py-2 text-white rounded-lg transition-all border border-gray-600 touch-manipulation ${extraClasses}`;
    return btn;
  }
}
