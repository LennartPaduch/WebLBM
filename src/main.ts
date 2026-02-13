import { showError } from "./ErrorHandler";
import { GPUController } from "./GPUController";
import { SimulationController } from "./SimulationController";
import { UIController } from "./UIController";

try {
  const gpu = await GPUController.create();

  const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  if (!canvas) throw new Error("Canvas element not found");
  gpu.configureCanvas(canvas);

  const limits = gpu.adapter.limits;
  const bytesPerCellInF = 9 * 2;
  const maxStorageBytes = Math.min(
    limits.maxBufferSize,
    limits.maxStorageBufferBindingSize,
  );
  const maxCellsByF = Math.floor((0.95 * maxStorageBytes) / bytesPerCellInF);
  const maxDim = Math.min(
    Math.floor(Math.sqrt(maxCellsByF)),
    limits.maxTextureDimension2D,
  );
  const MAX_CAP = 1 << 11;
  const isSmallViewport = window.matchMedia("(max-width: 900px)").matches;
  const isCoarsePointer = window.matchMedia("(pointer: coarse)").matches;
  const initialResolution = isSmallViewport || isCoarsePointer ? 1 << 9 : 1 << 10;

  const controller = new SimulationController(gpu, canvas, initialResolution);
  const ui = new UIController(controller, maxDim, MAX_CAP);
  await controller.recreate(initialResolution, maxDim, MAX_CAP, "Von Kármán Street");
  ui.syncAll();
} catch (e) {
  showError(e);
}
