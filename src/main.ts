import { showError } from "./ErrorHandler";
import { GPUController } from "./GPUController";
import { SimulationController } from "./SimulationController";
import { UIController } from "./UIController";

try {
  const gpu = await GPUController.create();

  // Find the canvas once
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  if (!canvas) throw new Error("Canvas element not found");
  gpu.configureCanvas(canvas);

  const limits = gpu.adapter.limits;
  const bytesPerCellInF = 9 * 2; // D2Q9 populations in f16 (SoA)
  const maxStorageBytes = Math.min(
    limits.maxBufferSize,
    limits.maxStorageBufferBindingSize,
  );
  const maxCellsByF = Math.floor((0.95 * maxStorageBytes) / bytesPerCellInF);
  const maxDim = Math.min(
    Math.floor(Math.sqrt(maxCellsByF)),
    limits.maxTextureDimension2D,
  );
  const MAX_CAP = 1 << 11; //2048
  const isSmallViewport = window.matchMedia("(max-width: 900px)").matches;
  const isCoarsePointer = window.matchMedia("(pointer: coarse)").matches;
  const initialResolution = isSmallViewport || isCoarsePointer ? 1 << 9 : 1 << 10; // 512 on mobile-ish devices, else 1024

  // Pass canvas to the controller
  const controller = new SimulationController(gpu, canvas, initialResolution);

  // Initialize UI
  const ui = new UIController(controller, maxDim, MAX_CAP);

  // Initial Start
  await controller.recreate(initialResolution, maxDim, MAX_CAP, "Von Kármán Street");
  ui.syncAll();
} catch (e) {
  showError(e);
}
