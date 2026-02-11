export class GPUController {
  adapter: GPUAdapter;
  device: GPUDevice;

  context!: GPUCanvasContext;
  contextFormat: GPUTextureFormat;

  private constructor(
    adapter: GPUAdapter,
    device: GPUDevice,
    format: GPUTextureFormat,
  ) {
    this.adapter = adapter;
    this.device = device;
    this.contextFormat = format;
  }

  static async create(): Promise<GPUController> {
    if (!navigator.gpu)
      throw new Error("WebGPU not supported on this browser.");

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No appropriate GPUAdapter found.");

    if (!adapter.features.has("shader-f16")) {
      throw new Error("Your hardware/browser does not support F16.");
    }

    const limits = adapter.limits;
    const device = await adapter.requestDevice({
      requiredFeatures: ["shader-f16"],
      requiredLimits: {
        maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
        maxBufferSize: limits.maxBufferSize,
        maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
        maxComputeWorkgroupSizeY: limits.maxComputeWorkgroupSizeY,
        maxComputeWorkgroupsPerDimension:
          limits.maxComputeWorkgroupsPerDimension,
        maxComputeInvocationsPerWorkgroup:
          limits.maxComputeInvocationsPerWorkgroup,
        maxTextureDimension2D: limits.maxTextureDimension2D,
      },
    });



    device.onuncapturederror = (e) => {
      console.error(`WebGPU uncaptured error: ${e.error?.message ?? e}`);
    };

    device.lost.then((info) => {
      console.error("WebGPU device lost:", info);
    });

    const format = navigator.gpu.getPreferredCanvasFormat();
    return new GPUController(adapter, device, format);
  }

  configureCanvas(canvas: HTMLCanvasElement) {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));

    canvas.width = w;
    canvas.height = h;

    const ctx = canvas.getContext("webgpu") as GPUCanvasContext | null;
    if (!ctx) throw new Error('Failed to get "webgpu" canvas context.');

    this.context = ctx;

    ctx.configure({
      device: this.device,
      format: this.contextFormat,
      alphaMode: "premultiplied",
    });
  }
}
