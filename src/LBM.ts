import type { GPUController } from "./GPUController";
import initWGSL from "./shader/init.wgsl?raw";
import stepWGSL from "./shader/step.wgsl?raw";
import blitWGSL from "./shader/blit.wgsl?raw";
import renderComputeWGSL from "./shader/render_compute.wgsl?raw";
import commonWgsl from "./shader/common.wgsl?raw";
import type { SceneBuildResult } from "./presets";

export const CELL = {
  FLUID: 0,
  SOLID: 1 << 0,
  EQ: 1 << 1,
} as const;

export const VisTypes = {
  VELOCITY: 0, // |u|
  VORTICITY: 1,
} as const;
export type VisType = (typeof VisTypes)[keyof typeof VisTypes];

export const VisColormaps = {
  VIRIDIS: 0,
  TURBO: 1,
  RdBu: 2,
} as const;
export type VisColormap = (typeof VisColormaps)[keyof typeof VisColormaps];

interface VisSettings {
  minValue: number;
  maxValue: number;
  type: VisType;
  colorMap: VisColormap;
}

const UNIFORM_SIZE = 256;

export class LBM {
  // grid
  #Nx: number;
  #Ny: number;
  #Q = 9;
  #cellCount: number;
  #fluidCellCount: number;

  // physics
  #tau = 0.6;
  #omega = 1.0 / this.#tau;
  #inletUx = 0.02;
  #inletUy = 0;

  #visSettings: VisSettings = {
    minValue: 0.0,
    maxValue: 0.05,
    type: VisTypes.VELOCITY,
    colorMap: VisColormaps.TURBO,
  };

  #gpu: GPUController;

  // buffers
  #f!: GPUBuffer;
  #maskBuffer!: GPUBuffer; // u32 per cell
  #maskCPU!: Uint32Array;
  #u!: GPUBuffer;
  #rho!: GPUBuffer;

  #initUniform!: GPUBuffer;
  #stepUniform!: GPUBuffer;

  // parity uniforms (StepDynamic in WGSL; only parity used, padding unused)
  #parityBuf0!: GPUBuffer;
  #parityBuf1!: GPUBuffer;

  // pipelines
  #pipeInit!: GPUComputePipeline;
  #pipeStep!: GPUComputePipeline;

  // bind groups
  #bgInit!: GPUBindGroup;
  #bgStep0!: GPUBindGroup;
  #bgStep1!: GPUBindGroup;

  // rendering
  #visTex!: GPUTexture;
  #visView!: GPUTextureView;
  #visSampler!: GPUSampler;

  #visUniform!: GPUBuffer; // VisParams
  #pipeVis!: GPUComputePipeline;
  #pipeBlit!: GPURenderPipeline;

  #bgVis!: GPUBindGroup;

  #bgBlit!: GPUBindGroup;

  // Simulation State
  tick = 0;
  #isRunning = false;
  #rafId: number = NaN;

  // Performance / Auto-tuning
  #stepsPerFrame = 1;
  #lastFrameTime = 0;

  // cached dispatch sizes
  #workgroupsX = 0;
  #workgroupsY = 0;

  // OPT: avoid per-call allocations for uniforms
  #visAB = new ArrayBuffer(UNIFORM_SIZE);
  #visDV = new DataView(this.#visAB);
  #visDirty = true;

  #initAB = new ArrayBuffer(UNIFORM_SIZE);
  #initDV = new DataView(this.#initAB);

  #stepAB = new ArrayBuffer(UNIFORM_SIZE);
  #stepDV = new DataView(this.#stepAB);

  constructor(nx: number, ny: number, gpu: GPUController) {
    this.#Nx = nx;
    this.#Ny = ny;
    this.#cellCount = nx * ny;
    this.#fluidCellCount = this.#cellCount;
    this.#gpu = gpu;
  }

  init = async (): Promise<void> => {
    const device = this.#gpu.device;

    const lim = device.limits;
    const maxInv = lim.maxComputeInvocationsPerWorkgroup;

    const candidates: Array<[number, number]> = [
      [64, 4],
      [32, 8],
      [16, 16],
      [32, 4],
      [8, 32],
      [16, 8],
      [8, 16],
    ];

    let WGX = 8, WGY = 8; // safe fallback
    for (const [x, y] of candidates) {
      if (x <= lim.maxComputeWorkgroupSizeX &&
        y <= lim.maxComputeWorkgroupSizeY &&
        x * y <= maxInv) {
        WGX = x; WGY = y;
        break;
      }
    }

    this.#workgroupsX = Math.ceil(this.#Nx / WGX);
    this.#workgroupsY = Math.ceil(this.#Ny / WGY);

    const constants = { WGX, WGY, WGZ: 1 };

    // ---------- buffers ----------
    const elems = this.#Q * this.#cellCount;
    const bytesF = elems * 2; // f16
    const sizeF4 = (bytesF + 3) & ~3;

    this.#f = device.createBuffer({
      label: "f",
      size: sizeF4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    this.#maskBuffer = device.createBuffer({
      label: "mask",
      size: this.#cellCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.#initUniform = device.createBuffer({
      label: "init params",
      size: UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.#stepUniform = device.createBuffer({
      label: "constant step params",
      size: UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // u is 2*C f32 = 8*C bytes
    this.#u = device.createBuffer({
      label: "global u array",
      size: this.#cellCount * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // rho is C f32 = 4*C bytes
    this.#rho = device.createBuffer({
      label: "global rho array",
      size: this.#cellCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // --- parity uniforms (StepDynamic) ---
    this.#parityBuf0 = device.createBuffer({
      label: "parity 0 (StepDynamic)",
      size: UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.#parityBuf1 = device.createBuffer({
      label: "parity 1 (StepDynamic)",
      size: UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    device.queue.writeBuffer(this.#parityBuf0, 0, new Uint32Array([0]));
    device.queue.writeBuffer(this.#parityBuf1, 0, new Uint32Array([1]));

    // write init + step uniforms once
    this.#writeInitUniform({ rho0: 1, inletUx: this.#inletUx, inletUy: this.#inletUy });
    this.#writeStepUniform();

    // ---------- pipelines ----------
    const modInit = device.createShaderModule({
      label: "init.wgsl",
      code: commonWgsl + "\n" + initWGSL,
    });
    const modStep = device.createShaderModule({
      label: "step.wgsl",
      code: commonWgsl + "\n" + stepWGSL,
    });

    this.#pipeInit = device.createComputePipeline({
      label: "init pipeline",
      layout: "auto",
      compute: { module: modInit, entryPoint: "initialize", constants },
    });

    this.#pipeStep = device.createComputePipeline({
      label: "step pipeline",
      layout: "auto",
      compute: { module: modStep, entryPoint: "step", constants },
    });

    // ---------- bind groups ----------
    this.#bgInit = device.createBindGroup({
      label: "init BG",
      layout: this.#pipeInit.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#f } },
        { binding: 1, resource: { buffer: this.#maskBuffer } },
        { binding: 2, resource: { buffer: this.#initUniform } },
        { binding: 3, resource: { buffer: this.#u } },
        { binding: 4, resource: { buffer: this.#rho } },
      ],
    });

    const stepLayout = this.#pipeStep.getBindGroupLayout(0);
    const stepEntriesBase = [
      { binding: 0, resource: { buffer: this.#f } },
      { binding: 1, resource: { buffer: this.#stepUniform } },
      { binding: 2, resource: { buffer: this.#maskBuffer } },
      { binding: 3, resource: { buffer: this.#u } },
      { binding: 4, resource: { buffer: this.#rho } },
    ] as const;

    this.#bgStep0 = device.createBindGroup({
      label: "step BG parity 0",
      layout: stepLayout,
      entries: [...stepEntriesBase, { binding: 5, resource: { buffer: this.#parityBuf0 } }],
    });

    this.#bgStep1 = device.createBindGroup({
      label: "step BG parity 1",
      layout: stepLayout,
      entries: [...stepEntriesBase, { binding: 5, resource: { buffer: this.#parityBuf1 } }],
    });

    // ---------- visualization ----------
    this.#visTex = device.createTexture({
      label: "visTex",
      size: { width: this.#Nx, height: this.#Ny },
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.#visView = this.#visTex.createView();
    this.#visSampler = device.createSampler({ minFilter: "nearest", magFilter: "nearest" });

    this.#visUniform = device.createBuffer({
      label: "VisParams",
      size: UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // compile viz shaders
    const visModule = device.createShaderModule({
      label: "render_compute.wgsl",
      code: commonWgsl + "\n" + renderComputeWGSL,
    });
    const blitModule = device.createShaderModule({
      label: "blit.wgsl",
      code: blitWGSL,
    });

    this.#pipeVis = device.createComputePipeline({
      label: "vis pipeline",
      layout: "auto",
      compute: { module: visModule, entryPoint: "render", constants },
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

    // write initial vis params once
    this.#writeVisUniformIfDirty();


    const visLayout = this.#pipeVis.getBindGroupLayout(0);

    this.#bgVis = device.createBindGroup({
      label: "vis BG (macros)",
      layout: visLayout,
      entries: [
        { binding: 0, resource: { buffer: this.#u } },          // global_u
        { binding: 1, resource: { buffer: this.#maskBuffer } }, // mask
        { binding: 2, resource: { buffer: this.#visUniform } }, // VisParams
        { binding: 3, resource: this.#visView },                // outputTex
      ],
    });

    // blit bind group
    this.#bgBlit = device.createBindGroup({
      label: "blit BG",
      layout: this.#pipeBlit.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.#visView },
        { binding: 1, resource: this.#visSampler },
      ],
    });
  };

  setTau = (tau: number): void => {
    this.#tau = tau;
    this.#omega = 1 / this.#tau;
  }

  #resetSimulation = (): void => {
    cancelAnimationFrame(this.#rafId);

    const device = this.#gpu.device;
    const enc = device.createCommandEncoder({ label: "init enc" });

    const pass = enc.beginComputePass({ label: "init pass" });
    pass.setPipeline(this.#pipeInit);
    pass.setBindGroup(0, this.#bgInit);
    pass.dispatchWorkgroups(this.#workgroupsX, this.#workgroupsY, 1);
    pass.end();

    device.queue.submit([enc.finish()]);

    this.tick = 0;
    this.#stepsPerFrame = 1;
  };

  setVisType = (newType: VisType): void => {
    // 1) Reference flow speed for visualization scaling.
    // Use inlet speed magnitude with a floor so the initial at-rest state still has a usable range.
    const uMag = Math.sqrt(this.#inletUx ** 2 + this.#inletUy ** 2);
    const uRef = Math.max(uMag, 0.02);

    // 2) Choose display bounds.
    // These are heuristic visualization ranges, not strict physical limits.
    let min = 0.0;
    let max = 1.0;

    switch (newType) {
      case VisTypes.VELOCITY: {
        // For fully developed channel flow, centerline speed is ~1.5x mean inlet speed.
        // Obstacles can locally increase speed, so we leave extra headroom.
        min = 0.0;
        max = uRef * 2;
        break;
      }
      case VisTypes.VORTICITY: {
        // In lattice units, |omega_z| scales like U/L.
        // This factor is tuned for readable wakes across presets, not a universal physical bound.
        const vortLimit = uRef * 0.01;
        min = -vortLimit;
        max = vortLimit;
        break;
      }
      default: {
        min = 0.0;
        max = uRef * 2;
      }
    }

    // 3) Apply
    this.#visSettings.minValue = min;
    this.#visSettings.maxValue = max;
    this.#visSettings.type = newType;
    this.#visDirty = true;

    if (!this.#isRunning) this.#executeBatch(0);
  };

  setVisColormap = (newColormap: VisColormap): void => {
    this.#visSettings.colorMap = newColormap;
    this.#visDirty = true;
    if (!this.#isRunning) this.#executeBatch(0);
  };

  setMask(result: SceneBuildResult): void {
    if (result.mask.length !== this.#cellCount) throw new Error("Mask size mismatch");
    this.#maskCPU = result.mask;
    this.#fluidCellCount = this.#countFluidCells(result.mask);

    const q = this.#gpu.device.queue;
    q.writeBuffer(
      this.#maskBuffer,
      0,
      result.mask.buffer,
      result.mask.byteOffset,
      result.mask.byteLength
    );

    this.#inletUx = result.sim.inletUx;
    this.#inletUy = result.sim.inletUy ?? 0;
    this.setTau(result.sim.tau);

    this.#writeInitUniform(
      {
        rho0: 1,
        inletUx: this.#inletUx,
        inletUy: this.#inletUy
      });
    this.#writeStepUniform();
  }

  #renderFrame = (timestamp: number) => {
    if (!this.#isRunning) return;

    const dt = timestamp - this.#lastFrameTime;
    this.#lastFrameTime = timestamp;

    if (dt > 0) {
      if (dt < 12) {
        this.#stepsPerFrame = Math.min(this.#stepsPerFrame + 1, 100);
      } else if (dt > 17 && this.#stepsPerFrame > 1) {
        this.#stepsPerFrame = Math.max(1, Math.floor(this.#stepsPerFrame * 0.9));
      }
    }

    this.#executeBatch(this.#stepsPerFrame);
    this.#rafId = requestAnimationFrame(this.#renderFrame);
  };

  #executeBatch = (steps: number): void => {
    const device = this.#gpu.device;
    const enc = device.createCommandEncoder({ label: "LBM Batch" });

    // Compute pass: STEP (N times) + VIS (once)
    {
      const cPass = enc.beginComputePass({ label: "Sim+Vis" });

      // --- step ---
      cPass.setPipeline(this.#pipeStep);
      for (let i = 0; i < steps; i++) {
        const bg = (this.tick & 1) === 0 ? this.#bgStep0 : this.#bgStep1;
        cPass.setBindGroup(0, bg);
        cPass.dispatchWorkgroups(this.#workgroupsX, this.#workgroupsY);
        this.tick++;
      }

      // --- vis ---
      this.#writeVisUniformIfDirty();
      cPass.setPipeline(this.#pipeVis);
      cPass.setBindGroup(0, this.#bgVis);
      cPass.dispatchWorkgroups(this.#workgroupsX, this.#workgroupsY);

      cPass.end();
    }

    // Blit pass
    {
      const view = this.#gpu.context.getCurrentTexture().createView();
      const rPass = enc.beginRenderPass({
        label: "blit",
        colorAttachments: [
          {
            view,
            loadOp: "load", // fullscreen triangle overwrites everything
            storeOp: "store",
          },
        ],
      });

      rPass.setPipeline(this.#pipeBlit);
      rPass.setBindGroup(0, this.#bgBlit);
      rPass.draw(3);
      rPass.end();
    }

    device.queue.submit([enc.finish()]);
  };

  run = (numSteps?: number): void => {
    if (this.#isRunning) return;
    this.#isRunning = true;

    if (numSteps !== undefined) {
      this.#executeBatch(numSteps);
      this.#isRunning = false;
      return;
    }

    this.#lastFrameTime = performance.now();
    this.#rafId = requestAnimationFrame(this.#renderFrame);
  };

  togglePause = (): void => {
    if (this.#isRunning) this.#pause();
    else this.run();
  };

  #pause = (): void => {
    this.#isRunning = false;
    cancelAnimationFrame(this.#rafId);
  };

  restart = (): void => {
    this.#pause();
    this.#resetSimulation();
    this.run();
  };

  isRunning = (): boolean => this.#isRunning;

  getCellcount = (): number => this.#cellCount;
  getFluidCellcount = (): number => this.#fluidCellCount;

  #countFluidCells = (mask: Uint32Array): number => {
    let count = 0;
    for (let i = 0; i < mask.length; i++) {
      if ((mask[i] & CELL.SOLID) === 0) count++;
    }
    return count;
  };

  // ---------- uniform writers (no allocations) ----------
  #writeVisUniformIfDirty = (): void => {
    if (!this.#visDirty) return;
    this.#visDirty = false;

    const dv = this.#visDV;
    let o = 0;

    dv.setUint32(o, this.#Nx, true); o += 4;
    dv.setUint32(o, this.#Ny, true); o += 4;
    dv.setUint32(o, this.#cellCount, true); o += 4;
    dv.setUint32(o, this.#visSettings.type | 0, true); o += 4;
    dv.setUint32(o, this.#visSettings.colorMap | 0, true); o += 4;
    dv.setFloat32(o, this.#visSettings.minValue, true); o += 4;
    dv.setFloat32(o, this.#visSettings.maxValue, true); o += 4;

    this.#gpu.device.queue.writeBuffer(this.#visUniform, 0, this.#visAB);
  };

  #writeInitUniform = (opts: { rho0: number; inletUx: number; inletUy: number }): void => {
    // rho0 is retained for API compatibility; the init uniform currently uses inlet velocity only.
    const dv = this.#initDV;
    let o = 0;

    dv.setUint32(o, this.#Nx, true); o += 4;
    dv.setUint32(o, this.#Ny, true); o += 4;
    dv.setUint32(o, this.#Q, true); o += 4;
    dv.setFloat32(o, opts.inletUx, true); o += 4;
    dv.setFloat32(o, opts.inletUy, true);

    this.#gpu.device.queue.writeBuffer(this.#initUniform, 0, this.#initAB);
  };

  #writeStepUniform = (): void => {
    const dv = this.#stepDV;
    let o = 0;

    dv.setUint32(o, this.#Nx, true); o += 4;
    dv.setUint32(o, this.#Ny, true); o += 4;
    dv.setUint32(o, this.#cellCount, true); o += 4;
    dv.setUint32(o, this.#Q, true); o += 4;

    dv.setFloat32(o, 1.0 /* rhoIn */, true); o += 4;
    dv.setFloat32(o, this.#inletUx /* uInx */, true); o += 4;
    dv.setFloat32(o, this.#inletUy /* uIny */, true); o += 4;
    dv.setFloat32(o, 1.0 /* rhoOut */, true); o += 4;

    dv.setFloat32(o, this.#omega, true);

    this.#gpu.device.queue.writeBuffer(this.#stepUniform, 0, this.#stepAB);
  };

  applyMaskRows(rows: Array<{ y: number; x0: number; x1: number }>, value: number) {
    if (!rows.length) return;

    const perRow = new Map<number, Array<{ x0: number; x1: number }>>();
    for (const { y, x0, x1 } of rows) {
      if (y < 0 || y >= this.#Ny) continue;

      const a = Math.max(0, Math.min(this.#Nx - 1, Math.min(x0, x1)));
      const b = Math.max(0, Math.min(this.#Nx - 1, Math.max(x0, x1)));
      if (a > b) continue;

      let spans = perRow.get(y);
      if (!spans) {
        spans = [];
        perRow.set(y, spans);
      }
      spans.push({ x0: a, x1: b });
    }

    const q = this.#gpu.device.queue;

    for (const [y, spans] of perRow) {
      spans.sort((a, b) => a.x0 - b.x0);
      const rowBase = y * this.#Nx;

      const writeSpan = (x0: number, x1: number) => {
        for (let x = x0; x <= x1; x++) {
          const idx = rowBase + x;
          const prev = this.#maskCPU[idx];
          if (prev === value) continue;

          const wasFluid = (prev & CELL.SOLID) === 0;
          const nowFluid = (value & CELL.SOLID) === 0;
          if (wasFluid && !nowFluid) this.#fluidCellCount--;
          else if (!wasFluid && nowFluid) this.#fluidCellCount++;

          this.#maskCPU[idx] = value;
        }

        const start = rowBase + x0;
        const count = x1 - x0 + 1;

        q.writeBuffer(
          this.#maskBuffer,
          start * 4,
          this.#maskCPU.buffer,
          start * 4,
          count * 4,
        );
      };

      let cur = spans[0];
      for (let i = 1; i < spans.length; i++) {
        const next = spans[i];
        if (next.x0 <= cur.x1 + 1) {
          cur.x1 = Math.max(cur.x1, next.x1);
          continue;
        }
        writeSpan(cur.x0, cur.x1);
        cur = next;
      }
      writeSpan(cur.x0, cur.x1);
    }
  }

  destroy = (): void => {
    this.#isRunning = false;
    cancelAnimationFrame(this.#rafId);

    this.#f?.destroy();
    this.#maskBuffer?.destroy();
    this.#u?.destroy();
    this.#rho?.destroy();

    this.#visTex?.destroy();
    this.#visUniform?.destroy();

    this.#initUniform?.destroy();
    this.#stepUniform?.destroy();

    this.#parityBuf0?.destroy();
    this.#parityBuf1?.destroy();
  };
}
