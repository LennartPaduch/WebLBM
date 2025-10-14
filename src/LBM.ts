import type { GPUController } from "./GPUController";
import initWGSL from "./shader/init.wgsl";
import stepWGSL from "./shader/step.wgsl";
import blitWGSL from "./shader/blit.wgsl";
import renderComputeWGSL from "./shader/render_compute.wgsl";

export const CELL = {
  FLUID: 0,
  SOLID: 1 << 0,
  EQ: 1 << 1,
} as const;

export const VisTypes = {
  VELOCITY: 0, // |u|
  DENSITY: 1, // rho
} as const;
type VisType = (typeof VisTypes)[keyof typeof VisTypes];

export const VisColormaps = {
  TURBO: 1,
  VIRIDIS: 0,
} as const;
type VisColormap = (typeof VisColormaps)[keyof typeof VisColormaps];

interface VisSettings {
  minValue: number;
  maxValue: number;
  type: VisType;
  colorMap: VisColormap;
}

export class LBM {
  // grid
  #Nx: number;
  #Ny: number;
  #Q = 9;
  #cellCount: number;

  // physics
  #tau = 0.7;
  #omega = 1 / this.#tau;
  #inletUx = 0.05;
  #inletUy = 0;

  #WORKGROUP_SIZE = 32;

  #visSettings: VisSettings = {
    minValue: 0.0,
    maxValue: 0.05,
    type: VisTypes.VELOCITY,
    colorMap: VisColormaps.TURBO,
  };

  // gpu
  #gpu: GPUController;

  // buffers
  #f!: GPUBuffer;
  #mask!: GPUBuffer; // u32 per cell
  #maskCPU!: Uint32Array;
  #u!: GPUBuffer;
  #rho!: GPUBuffer;
  #initUniform!: GPUBuffer;
  #stepUniform!: GPUBuffer;
  #stepUniformUpdated!: GPUBuffer; // updated parity

  // pipelines
  #pipeInit!: GPUComputePipeline;
  #pipeStep!: GPUComputePipeline;

  // bind groups (prebuilt)
  #bgInit!: GPUBindGroup;
  #bgStep!: GPUBindGroup;

  //rendering
  #visTex!: GPUTexture;
  #visView!: GPUTextureView;
  #visSampler!: GPUSampler;

  #visUniform!: GPUBuffer; // VisParams
  #pipeVis!: GPUComputePipeline;
  #pipeBlit!: GPURenderPipeline;

  #bgVis!: GPUBindGroup;
  #bgBlit!: GPUBindGroup; // Samples #visTex

  // step toggle
  #tick = 0;
  #parity: 0 | 1 = 0;
  #isRunning: boolean = false;
  #rafId: number = NaN;

  constructor(nx: number, ny: number, gpu: GPUController) {
    this.#Nx = nx;
    this.#Ny = ny;
    this.#cellCount = nx * ny;
    this.#gpu = gpu;
  }

  init = async (): Promise<void> => {
    const device = this.#gpu.device;
    // ---------- buffers ----------

    const elems = this.#Q * this.#cellCount;
    const bytesF = elems * 2; // 2 bytes per f16
    const size4 = (bytesF + 3) & ~3; // round up to multiple of 4

    this.#f = device.createBuffer({
      label: "f",
      size: size4,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });
    this.#mask = device.createBuffer({
      label: "mask",
      size: this.#cellCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.#maskCPU = this.#getMaskData();
    device.queue.writeBuffer(this.#mask, 0, this.#maskCPU.buffer);

    // uniforms (256B each)
    this.#initUniform = device.createBuffer({
      label: "init params",
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.#stepUniform = device.createBuffer({
      label: "constant step params",
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.#stepUniformUpdated = device.createBuffer({
      label: "updated step params",
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.#writeInitUniform({
      rho0: 1,
      inletUX: this.#inletUx,
      inletUY: this.#inletUy,
    });

    this.#u = device.createBuffer({
      label: "global u array",
      size: this.#cellCount * 4, // 2*C*sizeof(f16) = C*4
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });
    this.#rho = device.createBuffer({
      label: "global rho array",
      size: this.#cellCount * 2,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });

    this.#writeStepUniform();

    // ---------- pipelines ----------
    const modInit = device.createShaderModule({
      label: "init.wgsl",
      code: initWGSL,
    });
    const modStep = device.createShaderModule({
      label: "step.wgsl",
      code: stepWGSL,
    });

    this.#pipeInit = device.createComputePipeline({
      label: "init pipeline",
      layout: "auto",
      compute: { module: modInit, entryPoint: "initialize" },
    });
    this.#pipeStep = device.createComputePipeline({
      label: "step pipeline",
      layout: "auto",
      compute: { module: modStep, entryPoint: "step" },
    });
    // ---------- bind groups ----------
    this.#bgInit = device.createBindGroup({
      label: "init BG",
      layout: this.#pipeInit.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#f } }, // f (read_write)
        { binding: 1, resource: { buffer: this.#mask } }, // mask (read)
        { binding: 2, resource: { buffer: this.#initUniform } },
        { binding: 3, resource: { buffer: this.#u } },
        { binding: 4, resource: { buffer: this.#rho } },
      ],
    });

    this.#writeStepUniform();

    // step
    this.#bgStep = device.createBindGroup({
      label: "step",
      layout: this.#pipeStep.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#f } }, // f read_write
        { binding: 1, resource: { buffer: this.#stepUniform } },
        { binding: 2, resource: { buffer: this.#mask } },
        { binding: 3, resource: { buffer: this.#u } },
        { binding: 4, resource: { buffer: this.#rho } },
        { binding: 5, resource: { buffer: this.#stepUniformUpdated } },
      ],
    });

    // ---- visualization texture (RGBA8) ----
    this.#visTex = device.createTexture({
      label: "visTex",
      size: { width: this.#Nx, height: this.#Ny },
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.#visView = this.#visTex.createView();
    this.#visSampler = device.createSampler({
      minFilter: "linear",
      magFilter: "linear",
    });

    // ---- VisParams uniform (256B) ----
    this.#visUniform = device.createBuffer({
      label: "VisParams",
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ---- compile viz shaders ----
    const visModule = device.createShaderModule({
      label: "render_compute",
      code: renderComputeWGSL,
    });
    const blitModule = device.createShaderModule({
      label: "blit",
      code: blitWGSL,
    });

    this.#pipeVis = device.createComputePipeline({
      label: "vis pipeline",
      layout: "auto",
      compute: { module: visModule, entryPoint: "render" },
    });

    this.#pipeBlit = device.createRenderPipeline({
      label: "blit pipeline",
      layout: "auto",
      vertex: { module: blitModule, entryPoint: "vs" },
      fragment: {
        module: blitModule,
        entryPoint: "fs",
        targets: [{ format: this.#gpu.contextFormat }],
      },
    });

    // ---- viz bind group ----
    this.#bgVis = device.createBindGroup({
      label: "vis bg",
      layout: this.#pipeVis.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#f } }, // f
        { binding: 1, resource: { buffer: this.#visUniform } }, // VisParams
        { binding: 2, resource: { buffer: this.#stepUniformUpdated } }, // updated parity
        { binding: 3, resource: { buffer: this.#mask } }, // mask
        { binding: 4, resource: this.#visView }, // storage texture
      ],
    });
    // ---- blit bind group (sample visTex to canvas) ----
    const blitLayout = this.#pipeBlit.getBindGroupLayout(0);
    this.#bgBlit = device.createBindGroup({
      label: "blit BG",
      layout: blitLayout,
      entries: [
        { binding: 0, resource: this.#visView }, // sampled texture view
        { binding: 1, resource: this.#visSampler },
      ],
    });

    this.#resetSimulation();
  };

  #resetSimulation = (): void => {
    cancelAnimationFrame(this.#rafId);
    const device = this.#gpu.device;

    // ---------- run the GPU init once ----------
    const enc = device.createCommandEncoder({ label: "init enc" });
    const pass = enc.beginComputePass({ label: "init pass" });
    pass.setPipeline(this.#pipeInit);
    pass.setBindGroup(0, this.#bgInit);
    pass.dispatchWorkgroups(
      Math.ceil(this.#Nx / this.#WORKGROUP_SIZE),
      Math.ceil(this.#Ny / this.#WORKGROUP_SIZE),
      1
    );
    pass.end();
    device.queue.submit([enc.finish()]);

    this.#parity = 0;
    this.#tick = 0;
  };

  setVisType = (newType: VisType): void => {
    if (newType === VisTypes.DENSITY) {
      this.#visSettings.minValue = 1.0;
      this.#visSettings.maxValue = 1.05;
    } else if (newType === VisTypes.VELOCITY) {
      this.#visSettings.minValue = 0;
      this.#visSettings.maxValue = 0.05;
    }
    this.#visSettings.type = newType;
  };

  setVisColormap = (newColormap: VisColormap): void => {
    this.#visSettings.colorMap = newColormap;
  };

  #getMaskData = (): Uint32Array => {
    const maskData = new Uint32Array(this.#cellCount).fill(CELL.FLUID);

    // walls (top & bottom)
    for (let x = 0; x < this.#Nx; x++) {
      maskData[0 * this.#Nx + x] = CELL.SOLID; // bottom wall (y=0)
      maskData[(this.#Ny - 1) * this.#Nx + x] = CELL.SOLID; // top wall   (y=Ny-1)
    }

    // inlet window geometry
    const INLET_HEIGHT = 300;

    // Centered window, clamped to [1, Ny-2] (avoid solid walls)
    let y0 = Math.floor(this.#Ny * 0.5 - INLET_HEIGHT * 0.5);
    let y1 = Math.floor(this.#Ny * 0.5 + INLET_HEIGHT * 0.5) - 1; // inclusive

    y0 = Math.max(1, y0);
    y1 = Math.min(this.#Ny - 2, y1);

    // Guard against tiny/invalid windows
    if (y1 < y0) {
      [y0, y1] = [1, Math.min(this.#Ny - 2, 1)];
    }

    // inlet: left column, y in [y0..y1]
    for (let y = y0; y <= y1; y++) {
      maskData[y * this.#Nx + 0] = CELL.EQ;
    }

    // make the rest of the left column solid (excluding walls at y=0, y=Ny-1)
    for (let y = 1; y < y0; y++) {
      maskData[y * this.#Nx + 0] = CELL.SOLID;
    }
    for (let y = y1 + 1; y <= this.#Ny - 2; y++) {
      maskData[y * this.#Nx + 0] = CELL.SOLID;
    }

    // outlet: right column, skip wall rows
    for (let y = 1; y <= this.#Ny - 2; y++) {
      maskData[y * this.#Nx + (this.#Nx - 1)] = CELL.EQ;
    }

    // obstacle: filled circle
    const cx = Math.floor(this.#Nx * 0.5); // center x
    const cy = Math.floor(this.#Ny * 0.5); // center y
    const r = Math.floor(Math.min(this.#Nx, this.#Ny) * 0.1); // radius
    const r2 = r * r;

    const y0c = Math.max(0, cy - r);
    const y1c = Math.min(this.#Ny - 1, cy + r);

    for (let y = y0c; y <= y1c; y++) {
      const dy = y - cy;
      const dy2 = dy * dy;
      // span of x for this scanline: fill between [cx - sqrt(r^2 - dy^2), cx + sqrt(...)]
      let span = Math.floor(Math.sqrt(r2 - dy2));
      if (isNaN(span)) continue; // outside circle (can happen if dy2 > r2 due to rounding)
      let xl = Math.max(0, cx - span);
      let xr = Math.min(this.#Nx - 1, cx + span);
      const off = y * this.#Nx;
      for (let x = xl; x <= xr; x++) {
        maskData[off + x] = CELL.SOLID;
      }
    }

    return maskData;
  };

  #stepOnce = (): void => {
    const device = this.#gpu.device;

    // Update step uniform with current parity
    this.#updateStepUniform(this.#parity);

    const enc = device.createCommandEncoder({ label: "LBM frame (EP)" });

    // 1) step
    {
      const p = enc.beginComputePass({ label: "step" });
      p.setPipeline(this.#pipeStep);
      p.setBindGroup(0, this.#bgStep);
      p.dispatchWorkgroups(
        Math.ceil(this.#Nx / this.#WORKGROUP_SIZE),
        Math.ceil(this.#Ny / this.#WORKGROUP_SIZE)
      );
      p.end();
    }

    // 2) visualize (read current f)
    this.#writeVisUniform({
      min: this.#visSettings.minValue,
      max: this.#visSettings.maxValue,
      mode: this.#visSettings.type,
      cmap: this.#visSettings.colorMap, // 0 Viridis, 1 Turbo
    });
    {
      const p = enc.beginComputePass({ label: "vis pass" });
      p.setPipeline(this.#pipeVis);
      p.setBindGroup(0, this.#bgVis);
      p.dispatchWorkgroups(
        Math.ceil(this.#Nx / this.#WORKGROUP_SIZE),
        Math.ceil(this.#Ny / this.#WORKGROUP_SIZE)
      );
      p.end();
    }

    // 3) blit
    {
      const view = this.#gpu.context.getCurrentTexture().createView();
      const rp = enc.beginRenderPass({
        label: "blit",
        colorAttachments: [
          {
            view,
            loadOp: "clear",
            storeOp: "store",
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
          },
        ],
      });
      rp.setPipeline(this.#pipeBlit);
      rp.setBindGroup(0, this.#bgBlit);
      rp.draw(3);
      rp.end();
    }

    device.queue.submit([enc.finish()]);

    // Flip EP parity
    this.#parity ^= 1;
    this.#tick++;
  };

  run = (numSteps?: number): void => {
    if (this.#isRunning) {
      this.#resetSimulation();
    } else {
      this.#isRunning = true;
    }

    if (numSteps !== undefined) {
      for (let i = 0; i < numSteps; i++) this.#stepOnce();
      return;
    }
    const tick = () => {
      this.#stepOnce();
      this.#rafId = requestAnimationFrame(tick);
    };
    this.#rafId = requestAnimationFrame(tick);
  };

  #writeVisUniform = (opts: {
    min: number;
    max: number;
    mode: number;
    cmap: number;
  }): void => {
    const dv = new DataView(new ArrayBuffer(256));
    let o = 0;
    dv.setUint32(o, this.#Nx, true);
    o += 4; // Nx
    dv.setUint32(o, this.#Ny, true);
    o += 4; // Ny
    dv.setUint32(o, this.#cellCount, true);
    o += 4; // cellCount
    dv.setUint32(o, opts.mode | 0, true);
    o += 4; // mode
    dv.setUint32(o, opts.cmap | 0, true);
    o += 4; // cmap
    dv.setFloat32(o, opts.min, true);
    o += 4; // vmin
    dv.setFloat32(o, opts.max, true);
    o += 4; // vmax
    //o += 4; // _pad0
    this.#gpu.device.queue.writeBuffer(this.#visUniform, 0, dv.buffer);
  };

  // ---------- uniforms writers ----------
  #writeInitUniform = (opts: {
    rho0: number;
    inletUX: number;
    inletUY: number;
  }): void => {
    const dv = new DataView(new ArrayBuffer(256));
    let o = 0;
    dv.setUint32(o, this.#Nx, true);
    o += 4;
    dv.setUint32(o, this.#Ny, true);
    o += 4;
    dv.setUint32(o, this.#Q, true);
    o += 4;
    dv.setFloat32(o, opts.inletUX, true);
    o += 4;
    dv.setFloat32(o, opts.inletUY, true);

    this.#gpu.device.queue.writeBuffer(this.#initUniform, 0, dv.buffer);
  };

  #writeStepUniform = (): void => {
    const dv = new DataView(new ArrayBuffer(256));
    let o = 0;
    dv.setUint32(o, this.#Nx, true);
    o += 4;
    dv.setUint32(o, this.#Ny, true);
    o += 4;
    dv.setUint32(o, this.#cellCount, true);
    o += 4;
    dv.setUint32(o, this.#Q, true);
    o += 4;

    dv.setFloat32(o, 1.0 /* rhoIn */, true);
    o += 4;
    dv.setFloat32(o, this.#inletUx /* uInx */, true);
    o += 4;
    dv.setFloat32(o, this.#inletUy /* uIny */, true);
    o += 4;
    dv.setFloat32(o, 1.0 /* rhoOut*/, true);
    o += 4;

    dv.setFloat32(o, this.#omega, true);
    // o += 4;
    // o += 12; // _pad0, _pad1, _pad2

    this.#gpu.device.queue.writeBuffer(this.#stepUniform, 0, dv.buffer);
  };

  #updateStepUniform = (parity: 0 | 1): void => {
    const dv = new DataView(new ArrayBuffer(16));
    dv.setUint32(0, parity, true);
    this.#gpu.device.queue.writeBuffer(this.#stepUniformUpdated, 0, dv.buffer);
  };

  resetMask = (): void => {
    this.#maskCPU = this.#getMaskData();
    this.#gpu.device.queue.writeBuffer(this.#mask, 0, this.#maskCPU.buffer);
  };

  applyMaskRows(
    rows: Array<{ y: number; x0: number; x1: number }>,
    value: number
  ) {
    if (!rows.length) return;

    // merge per row to minimize GPU writes
    const perRow = new Map<number, { x0: number; x1: number }>();
    for (const { y, x0, x1 } of rows) {
      const r = perRow.get(y);
      if (!r) perRow.set(y, { x0: Math.min(x0, x1), x1: Math.max(x0, x1) });
      else {
        r.x0 = Math.min(r.x0, x0);
        r.x1 = Math.max(r.x1, x1);
      }
    }

    for (const [y, seg] of perRow) {
      const { x0, x1 } = seg;
      const off = y * this.#Nx;
      for (let x = x0; x <= x1; x++) this.#maskCPU[off + x] = value;

      // start = y * Nx + x0
      // count = x1 - x0 + 1
      const start = y * this.#Nx + x0;
      const count = x1 - x0 + 1;

      const dstByteOffset = start * 4; // u32 -> 4 bytes
      const srcByteOffset = start * 4; // where the span starts in the CPU buffer
      const byteSize = count * 4;

      this.#gpu.device.queue.writeBuffer(
        this.#mask, // destination GPUBuffer
        dstByteOffset, // where to write in GPU buffer (bytes)
        this.#maskCPU.buffer, // source ArrayBuffer
        srcByteOffset, // where to read from (bytes)
        byteSize // how many bytes to copy
      );
    }
  }
}
