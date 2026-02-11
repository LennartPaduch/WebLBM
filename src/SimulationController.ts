import { CanvasPainter } from "./canvas";
import type { GPUController } from "./GPUController";
import { LBM, type VisType, type VisColormap } from "./LBM";
import { makeBuildContext, Presets, resolveParams } from "./presets";

export class SimulationController {
  #gpu: GPUController;
  #canvas: HTMLCanvasElement;

  lbm!: LBM;
  painter!: CanvasPainter;
  N: number;
  currentPresetName = "";

  constructor(gpu: GPUController, canvas: HTMLCanvasElement, initialN: number) {
    this.#gpu = gpu;
    this.#canvas = canvas;
    this.N = initialN;
  }

  async recreate(
    NxNy: number,
    maxDim: number,
    maxCap: number,
    currentPresetName: string,
  ) {
    this.currentPresetName = currentPresetName;

    // 1. Hardware Clamp
    const safeN = Math.min(NxNy, maxDim, maxCap);
    this.N = safeN;

    // 2. Cleanup
    this.painter?.destroy();
    this.lbm?.destroy();

    // 3. Allocation (using the stored #gpu and safe N)
    this.lbm = new LBM(this.N, this.N, this.#gpu);
    await this.lbm.init();

    // 4. Painter (using the stored #canvas reference)
    this.painter = new CanvasPainter({
      canvas: this.#canvas,
      Nx: this.N,
      Ny: this.N,
      onPaint: (rows, value) => this.lbm.applyMaskRows(rows, value),
    });
    this.painter.enable();

    // 5. Apply Preset settings
    const preset = Presets[currentPresetName];
    if (preset) {
      // params: defaults + (optional) overrides from UI later
      const params = resolveParams(preset /*, overrides */);

      // build context (seeded by preset name so random presets are stable)
      const ctx = makeBuildContext(this.N, this.N, currentPresetName, params);

      const result = preset.build(ctx);

      // apply mask
      this.lbm.setMask(result);

      // apply visualization after sim params (inlet/tau) are set
      const visType = result.visType ?? preset.visType;
      const colormap = result.colormap ?? preset.colormap;
      if (visType !== undefined) this.lbm.setVisType(visType as VisType);
      if (colormap !== undefined) this.lbm.setVisColormap(colormap as VisColormap);

    }

    // 6. Ignition
    this.lbm.restart();
  }
}
