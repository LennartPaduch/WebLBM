export class GPUController {
  adapter: GPUAdapter;
  device: GPUDevice;

  context: GPUCanvasContext;
  contextFormat: GPUTextureFormat;

  private constructor(
    adapter: GPUAdapter,
    device: GPUDevice,
    context: GPUCanvasContext,
    format: GPUTextureFormat
  ) {
    this.adapter = adapter;
    this.device = device;
    this.context = context;
    this.contextFormat = format;
  }
  static async create(): Promise<GPUController> {
    if (!navigator.gpu) {
      throw new Error("WebGPU not supported on this browser.");
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("No appropriate GPUAdapter found.");
    }

    const canF16 = adapter?.features.has("shader-f16") ?? false;
    if (!canF16) {
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

    // canvas
    const canvas = document.querySelector("canvas");
    if (!canvas) throw new Error("No <canvas> element found.");
    const ctx = canvas.getContext("webgpu") as GPUCanvasContext | null;
    if (!ctx) throw new Error('Failed to get "webgpu" canvas context.');

    const format = navigator.gpu.getPreferredCanvasFormat();

    try {
      ctx.configure({ device, format, alphaMode: "premultiplied" });
    } catch (err) {
      throw new Error("Failed to configure WebGPU canvas context.");
    }

    return new GPUController(adapter, device, ctx, format);
  }

  createBuffer(
    label: string,
    byteSize: number,
    usage: GPUBufferUsageFlags
  ): GPUBuffer {
    return this.device.createBuffer({ label, size: byteSize, usage });
  }

  createBufferWithData(
    label: string,
    data: ArrayBufferView,
    usage: GPUBufferUsageFlags
  ): GPUBuffer {
    const buf = this.device.createBuffer({
      label,
      size: data.byteLength,
      usage: usage | GPUBufferUsage.COPY_DST,
      mappedAtCreation: false,
    });
    this.device.queue.writeBuffer(
      buf,
      0,
      data.buffer,
      data.byteOffset,
      data.byteLength
    );
    return buf;
  }

  getPreferredCanvasFormat = (): GPUTextureFormat => {
    return navigator.gpu.getPreferredCanvasFormat();
  };
}
