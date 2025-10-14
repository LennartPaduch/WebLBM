export class GPUController {
  adapter: GPUAdapter;
  device: GPUDevice;

  context: GPUCanvasContext;
  contextFormat: GPUTextureFormat;

  private constructor(adapter: GPUAdapter, device: GPUDevice) {
    this.adapter = adapter;
    this.device = device;

    const canvas = document.querySelector("canvas")!;
    const context = canvas.getContext("webgpu") as GPUCanvasContext;

    const contextFormat = navigator.gpu.getPreferredCanvasFormat();

    // Configure the canvas
    context.configure({
      device,
      format: contextFormat,
      alphaMode: "premultiplied",
    });

    // Configure the canvas
    context.configure({
      device,
      format: contextFormat,
      alphaMode: "premultiplied",
    });

    this.context = context;
    this.contextFormat = contextFormat;
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
      requiredFeatures: canF16 ? ["shader-f16"] : [],
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

    return new GPUController(adapter, device);
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
