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

    const safeN = Math.min(NxNy, maxDim, maxCap);
    this.N = safeN;

    this.painter?.destroy();
    this.lbm?.destroy();

    this.lbm = new LBM(this.N, this.N, this.#gpu);
    await this.lbm.init();

    this.painter = new CanvasPainter({
      canvas: this.#canvas,
      Nx: this.N,
      Ny: this.N,
      onPaint: (rows, value) => this.lbm.applyMaskRows(rows, value),
    });
    this.painter.enable();

    const preset = Presets[currentPresetName];
    if (preset) {
      const params = resolveParams(preset);
      const ctx = makeBuildContext(this.N, this.N, currentPresetName, params);

      const result = preset.build(ctx);
      this.lbm.setMask(result);
      const visType = result.visType ?? preset.visType;
      const colormap = result.colormap ?? preset.colormap;
      if (visType !== undefined) this.lbm.setVisType(visType as VisType);
      if (colormap !== undefined) this.lbm.setVisColormap(colormap as VisColormap);
    }

    this.lbm.restart();
  }
}
