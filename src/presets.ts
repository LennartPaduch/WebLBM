import { CELL, VisTypes, VisColormaps, type VisType, type VisColormap } from "./LBM";

export type Side = "left" | "right" | "top" | "bottom";

export type NumberParam = {
  kind: "number";
  label: string;
  min: number;
  max: number;
  step?: number;
  default: number;
};

export type BoolParam = {
  kind: "bool";
  label: string;
  default: boolean;
};

export type EnumParam<T extends string | number> = {
  kind: "enum";
  label: string;
  options: readonly { label: string; value: T }[];
  default: T;
};

export type ParamSpec = NumberParam | BoolParam | EnumParam<any>;
export type ParamValues = Record<string, number | boolean | string>;

export interface SceneBuildResult {
  mask: Uint32Array;

  visType?: VisType;
  colormap?: VisColormap;

  sim: {
    inletUx: number;
    inletUy?: number;
    tau: number;
  };
}

export interface BuildContext {
  nx: number;
  ny: number;
  rng: Rng;
  params: ParamValues;

  X: (u: Unit) => number;
  Y: (u: Unit) => number;
  W: (u: Unit) => number;
  H: (u: Unit) => number;
}

export interface ScenePreset {
  description?: string;
  resolution?: number;

  visType?: VisType;
  colormap?: VisColormap;

  params?: Record<string, ParamSpec>;

  build: (ctx: BuildContext) => SceneBuildResult;
}

export function resolveParams(preset: ScenePreset, overrides: ParamValues = {}): ParamValues {
  const out: ParamValues = {};
  const specs = preset.params ?? {};
  for (const [k, spec] of Object.entries(specs)) out[k] = spec.default;
  for (const [k, v] of Object.entries(overrides)) out[k] = v;
  return out;
}
export type Unit = number | { rel: number };
export const rel = (v: number): Unit => ({ rel: v });

function toPix(n: number, u: Unit): number {
  if (typeof u === "number") return Math.round(u);
  return Math.round(u.rel * (n - 1));
}
function toLen(n: number, u: Unit): number {
  if (typeof u === "number") return Math.max(1, Math.round(u));
  return Math.max(1, Math.round(u.rel * n));
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function isSolidCell(v: number) {
  return (v & CELL.SOLID) !== 0;
}

function finalizeChannelBoundaries(b: MaskBuilder) {
  addTopBottomWalls(b);

  // EQ only where the adjacent interior cell is fluid.
  for (let y = 1; y <= b.ny - 2; y++) {
    const open = !isSolidCell(b.mask[y * b.nx + 1]);
    b.set(0, y, open ? CELL.EQ : CELL.SOLID);
  }

  // EQ only where the adjacent interior cell is fluid.
  for (let y = 1; y <= b.ny - 2; y++) {
    const open = !isSolidCell(b.mask[y * b.nx + (b.nx - 2)]);
    b.set(b.nx - 1, y, open ? CELL.EQ : CELL.SOLID);
  }
}
function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function easeCos(t: number) {
  t = Math.max(0, Math.min(1, t));
  // Smooth 0..1 ramp.
  return 0.5 - 0.5 * Math.cos(Math.PI * t);
}

function solidifyOutsideBand(b: MaskBuilder, halfHeightAtX: (x: number) => number) {
  const midY = Math.floor(b.ny * 0.5);

  for (let x = 1; x <= b.nx - 2; x++) {
    const hh = Math.max(2, Math.min(Math.floor((b.ny - 3) * 0.5), halfHeightAtX(x)));
    const yLo = Math.max(1, midY - hh);
    const yHi = Math.min(b.ny - 2, midY + hh);

    if (yLo > 1) b.fillRect(x, 1, 1, yLo - 1, CELL.SOLID);
    if (yHi < b.ny - 2) b.fillRect(x, yHi + 1, 1, (b.ny - 2) - yHi, CELL.SOLID);
  }
}

export interface Rng {
  next(): number;
  int(min: number, maxInclusive: number): number;
}

function hashStringToSeed(s: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mulberry32(seed: number): Rng {
  let t = seed >>> 0;
  return {
    next() {
      t += 0x6d2b79f5;
      let x = t;
      x = Math.imul(x ^ (x >>> 15), x | 1);
      x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
      return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    },
    int(min: number, maxInclusive: number) {
      const r = this.next();
      return min + Math.floor(r * (maxInclusive - min + 1));
    },
  };
}

export function makeBuildContext(
  nx: number,
  ny: number,
  presetNameOrSeed: string | number,
  params: ParamValues
): BuildContext {
  const seed = typeof presetNameOrSeed === "number" ? presetNameOrSeed : hashStringToSeed(presetNameOrSeed);
  const rng = mulberry32(seed);

  return {
    nx,
    ny,
    rng,
    params,
    X: (u) => clamp(toPix(nx, u), 0, nx - 1),
    Y: (u) => clamp(toPix(ny, u), 0, ny - 1),
    W: (u) => clamp(toLen(nx, u), 1, nx),
    H: (u) => clamp(toLen(ny, u), 1, ny),
  };
}

class MaskBuilder {
  readonly nx: number;
  readonly ny: number;
  readonly mask: Uint32Array;

  constructor(nx: number, ny: number, fill: number = CELL.FLUID) {
    this.nx = nx;
    this.ny = ny;
    this.mask = new Uint32Array(nx * ny).fill(fill);
  }

  #idx(x: number, y: number) {
    return y * this.nx + x;
  }

  set(x: number, y: number, v: number) {
    if (x < 0 || x >= this.nx || y < 0 || y >= this.ny) return;
    this.mask[this.#idx(x, y)] = v;
  }

  fillRect(x0: number, y0: number, w: number, h: number, v: number) {
    const x1 = x0 + w - 1;
    const y1 = y0 + h - 1;
    for (let y = Math.max(0, y0); y <= Math.min(this.ny - 1, y1); y++) {
      const row = y * this.nx;
      for (let x = Math.max(0, x0); x <= Math.min(this.nx - 1, x1); x++) {
        this.mask[row + x] = v;
      }
    }
  }

  fillCircle(cx: number, cy: number, r: number, v: number) {
    const r2 = r * r;
    const y0 = Math.max(0, cy - r);
    const y1 = Math.min(this.ny - 1, cy + r);
    for (let y = y0; y <= y1; y++) {
      const dy = y - cy;
      const span = Math.floor(Math.sqrt(Math.max(0, r2 - dy * dy)));
      const x0 = Math.max(0, cx - span);
      const x1 = Math.min(this.nx - 1, cx + span);
      const row = y * this.nx;
      for (let x = x0; x <= x1; x++) this.mask[row + x] = v;
    }
  }

  line(x0: number, y0: number, x1: number, y1: number, v: number) {
    // Bresenham integer raster.
    let dx = Math.abs(x1 - x0);
    let sx = x0 < x1 ? 1 : -1;
    let dy = -Math.abs(y1 - y0);
    let sy = y0 < y1 ? 1 : -1;
    let err = dx + dy;

    let x = x0, y = y0;
    while (true) {
      this.set(x, y, v);
      if (x === x1 && y === y1) break;
      const e2 = 2 * err;
      if (e2 >= dy) { err += dy; x += sx; }
      if (e2 <= dx) { err += dx; y += sy; }
    }
  }

  fillPolygon(points: Array<{ x: number; y: number }>, v: number) {
    // Ray-casting fill over polygon bounding box.
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of points) {
      minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
    }
    minX = clamp(Math.floor(minX), 0, this.nx - 1);
    maxX = clamp(Math.ceil(maxX), 0, this.nx - 1);
    minY = clamp(Math.floor(minY), 0, this.ny - 1);
    maxY = clamp(Math.ceil(maxY), 0, this.ny - 1);

    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        if (pointInPoly(x + 0.5, y + 0.5, points)) this.set(x, y, v);
      }
    }
  }
}

function pointInPoly(px: number, py: number, poly: Array<{ x: number; y: number }>): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x, yi = poly[i].y;
    const xj = poly[j].x, yj = poly[j].y;
    const intersect =
      (yi > py) !== (yj > py) &&
      px < ((xj - xi) * (py - yi)) / (yj - yi + 1e-12) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function addTopBottomWalls(b: MaskBuilder) {
  // No-slip top and bottom walls.
  for (let x = 0; x < b.nx; x++) {
    b.set(x, 0, CELL.SOLID);
    b.set(x, b.ny - 1, CELL.SOLID);
  }
}

function setSide(b: MaskBuilder, side: Side, v: number, skipCorners = true) {
  const y0 = skipCorners ? 1 : 0;
  const y1 = skipCorners ? b.ny - 2 : b.ny - 1;
  const x0 = skipCorners ? 1 : 0;
  const x1 = skipCorners ? b.nx - 2 : b.nx - 1;

  if (side === "left") for (let y = y0; y <= y1; y++) b.set(0, y, v);
  if (side === "right") for (let y = y0; y <= y1; y++) b.set(b.nx - 1, y, v);
  if (side === "bottom") for (let x = x0; x <= x1; x++) b.set(x, 0, v);
  if (side === "top") for (let x = x0; x <= x1; x++) b.set(x, b.ny - 1, v);
}

function addInletWindowLeft(
  b: MaskBuilder,
  y0: number,
  y1: number,
  inletType: number = CELL.EQ,
  restType: number = CELL.SOLID
) {
  // Keep inlet away from fixed wall rows.
  y0 = clamp(y0, 1, b.ny - 2);
  y1 = clamp(y1, 1, b.ny - 2);
  if (y1 < y0) [y0, y1] = [y1, y0];

  for (let y = 1; y <= b.ny - 2; y++) b.set(0, y, restType);
  for (let y = y0; y <= y1; y++) b.set(0, y, inletType);
}

function addOutletRight(b: MaskBuilder, outletType: number = CELL.EQ) {
  setSide(b, "right", outletType, true);
}

function makeDefaultChannel(b: MaskBuilder, inletWindowRelHeight = 1.0) {
  addTopBottomWalls(b);

  // Centered inlet window on the left boundary.
  const winH = Math.max(1, Math.floor((b.ny - 2) * inletWindowRelHeight));
  let y0 = Math.floor((b.ny - 1) * 0.5 - winH * 0.5);
  let y1 = y0 + winH - 1;
  y0 = clamp(y0, 1, b.ny - 2);
  y1 = clamp(y1, 1, b.ny - 2);

  addInletWindowLeft(b, y0, y1, CELL.EQ, CELL.SOLID);
  addOutletRight(b, CELL.EQ);
}

export const Presets: Record<string, ScenePreset> = {
  "Empty Tunnel": {
    description: "Classic channel: solid top/bottom, inlet window on the left, outlet on the right.",
    params: {
      inletWindow: { kind: "number", label: "Inlet window (rel height)", min: 0.1, max: 1.0, step: 0.05, default: 1.0 },
    },
    build: ({ nx, ny, params }) => {
      const b = new MaskBuilder(nx, ny, CELL.FLUID);
      makeDefaultChannel(b, Number(params.inletWindow));
      return {
        mask: b.mask,
        sim: { inletUx: 0.05, tau: 0.6, }
      };
    },
  },

  "Von Kármán Street": {
    description: "Flow past a cylinder. Great with vorticity visualization.",
    visType: VisTypes.VORTICITY,
    colormap: VisColormaps.RdBu,
    params: {
      radius: { kind: "number", label: "Cylinder radius (rel Ny)", min: 0.02, max: 0.12, step: 0.005, default: 0.04 },
      xPos: { kind: "number", label: "Cylinder x (rel Nx)", min: 0.05, max: 0.35, step: 0.01, default: 0.15 },
      yPos: { kind: "number", label: "Cylinder y (rel Ny)", min: 0.15, max: 0.85, step: 0.01, default: 0.47 },
    },
    build: ({ nx, ny, params }) => {
      const b = new MaskBuilder(nx, ny, CELL.FLUID);
      makeDefaultChannel(b, 1.0);

      const cx = Math.floor(nx * Number(params.xPos));
      const cy = Math.floor(ny * Number(params.yPos));
      const r = Math.max(2, Math.floor(ny * Number(params.radius)));
      b.fillCircle(cx, cy, r, CELL.SOLID);

      return {
        mask: b.mask,
        visType: VisTypes.VORTICITY,
        colormap: VisColormaps.RdBu,
        sim: { inletUx: 0.06, tau: 0.57 }
      };
    },
  },

  "Staggered Grid": {
    description: "A repeated obstacle array. Nice for seeing wakes and mixing.",
    params: {
      r: { kind: "number", label: "Obstacle radius (px)", min: 2, max: 24, step: 1, default: 8 },
      rows: { kind: "number", label: "Rows", min: 2, max: 8, step: 1, default: 4 },
      cols: { kind: "number", label: "Cols", min: 2, max: 10, step: 1, default: 4 },
      x0: { kind: "number", label: "Start x (rel Nx)", min: 0.15, max: 0.5, step: 0.01, default: 0.3 },
      x1: { kind: "number", label: "End x (rel Nx)", min: 0.35, max: 0.8, step: 0.01, default: 0.6 },
      y0: { kind: "number", label: "Start y (rel Ny)", min: 0.1, max: 0.4, step: 0.01, default: 0.2 },
      y1: { kind: "number", label: "End y (rel Ny)", min: 0.6, max: 0.9, step: 0.01, default: 0.9 },
    },
    build: ({ nx, ny, params }) => {
      const b = new MaskBuilder(nx, ny, CELL.FLUID);
      makeDefaultChannel(b, 1.0);

      const r = Math.floor(Number(params.r));
      const rows = Math.floor(Number(params.rows));
      const cols = Math.floor(Number(params.cols));

      const x0 = Math.floor(nx * Number(params.x0));
      const x1 = Math.floor(nx * Number(params.x1));
      const y0 = Math.floor(ny * Number(params.y0));
      const y1 = Math.floor(ny * Number(params.y1));

      for (let j = 0; j < rows; j++) {
        const tY = rows === 1 ? 0.5 : j / (rows - 1);
        const y = Math.round(y0 + tY * (y1 - y0));
        for (let i = 0; i < cols; i++) {
          const tX = cols === 1 ? 0.5 : i / (cols - 1);
          let x = Math.round(x0 + tX * (x1 - x0));
          if (j % 2 === 1) x += Math.round(0.5 * (x1 - x0) / Math.max(1, cols - 1));
          b.fillCircle(clamp(x, 1, nx - 2), clamp(y, 1, ny - 2), r, CELL.SOLID);
        }
      }

      return {
        mask: b.mask,
        sim: { inletUx: 0.03, tau: 0.65 }
      };
    },
  },

  "Backward-Facing Step": {
    description: "A classic benchmark: sudden expansion after a step.",
    visType: VisTypes.VELOCITY,
    colormap: VisColormaps.TURBO,
    params: {
      stepX: { kind: "number", label: "Step start x (rel Nx)", min: 0.05, max: 0.4, step: 0.01, default: 0.35 },
      stepH: { kind: "number", label: "Step height (rel Ny)", min: 0.05, max: 0.45, step: 0.01, default: 0.25 },
      inletWindow: { kind: "number", label: "Inlet window (rel height)", min: 0.1, max: 1.0, step: 0.05, default: 0.6 },
    },
    build: ({ nx, ny, params }) => {
      const b = new MaskBuilder(nx, ny, CELL.FLUID);
      makeDefaultChannel(b, Number(params.inletWindow));
      
      const xStep = Math.floor(nx * Number(params.stepX));
      const h = Math.floor((ny - 2) * Number(params.stepH));
      // Bottom step obstacle.
      b.fillRect(1, 1, clamp(xStep, 1, nx - 2), clamp(h, 1, ny - 2), CELL.SOLID);

      return {
        mask: b.mask,
        visType: VisTypes.VELOCITY,
        colormap: VisColormaps.TURBO,
        sim: { inletUx: 0.04, tau: 0.62 }
      };
    },
  },

  "Venturi (Nozzle)": {
    description: "Wide→converge→throat→diverge→wide. Smooth walls + throat length.",
    visType: VisTypes.VELOCITY,
    colormap: VisColormaps.TURBO,
    params: {
      inletH: { kind: "number", label: "Inlet half-height (rel Ny)", min: 0.15, max: 0.49, step: 0.01, default: 0.40 },
      throatH: { kind: "number", label: "Throat half-height (rel Ny)", min: 0.05, max: 0.35, step: 0.01, default: 0.15 },
      throatX: { kind: "number", label: "Throat center x (rel Nx)", min: 0.25, max: 0.75, step: 0.01, default: 0.25 },
      convLen: { kind: "number", label: "Converge length (rel Nx)", min: 0.05, max: 0.35, step: 0.01, default: 0.35 },
      throatLen: { kind: "number", label: "Throat length (rel Nx)", min: 0.00, max: 0.20, step: 0.01, default: 0.06 },
      divLen: { kind: "number", label: "Diverge length (rel Nx)", min: 0.05, max: 0.45, step: 0.01, default: 0.28 },
    },
    build: ({ nx, ny, params }) => {
      const b = new MaskBuilder(nx, ny, CELL.FLUID);

      const inletHalf = Math.floor((ny - 2) * Number(params.inletH));
      const throatHalf = Math.floor((ny - 2) * Number(params.throatH));

      const xC = Math.floor((nx - 1) * Number(params.throatX));
      const convLen = Math.floor((nx - 2) * Number(params.convLen));
      const throatLen = Math.floor((nx - 2) * Number(params.throatLen));
      const divLen = Math.floor((nx - 2) * Number(params.divLen));

      const xTh0 = clamp(xC - Math.floor(throatLen * 0.5), 2, nx - 3);
      const xTh1 = clamp(xTh0 + throatLen, 2, nx - 3);

      const xCv0 = clamp(xTh0 - convLen, 1, xTh0);
      const xDv1 = clamp(xTh1 + divLen, xTh1, nx - 2);

      const halfHeightAtX = (x: number) => {
        if (x < xCv0) return inletHalf;

        if (x < xTh0) {
          const t = easeCos((x - xCv0) / Math.max(1, (xTh0 - xCv0)));
          return Math.floor(lerp(inletHalf, throatHalf, t));
        }

        if (x <= xTh1) return throatHalf;

        if (x <= xDv1) {
          const t = easeCos((x - xTh1) / Math.max(1, (xDv1 - xTh1)));
          return Math.floor(lerp(throatHalf, inletHalf, t));
        }

        return inletHalf;
      };
      
      solidifyOutsideBand(b, halfHeightAtX);

      // Apply BCs after geometry so openings match channel walls.
      finalizeChannelBoundaries(b);

      return {
        mask: b.mask,
        visType: VisTypes.VELOCITY,
        colormap: VisColormaps.TURBO,
        sim: { inletUx: 0.02, tau: 0.63 }
      };
    },
  },

  "Porous Media (Random Disks)": {
    description: "Random obstacle field (deterministic via seed). Great for pressure drop / tortuosity vibes.",
    visType: VisTypes.VORTICITY,
    colormap: VisColormaps.RdBu,
    params: {
      seed: { kind: "number", label: "Seed", min: 0, max: 10_000, step: 1, default: 1 },
      count: { kind: "number", label: "Obstacle count", min: 10, max: 400, step: 5, default: 120 },
      rMin: { kind: "number", label: "Min radius (px)", min: 2, max: 30, step: 1, default: 3 },
      rMax: { kind: "number", label: "Max radius (px)", min: 2, max: 60, step: 1, default: 10 },
      xPad: { kind: "number", label: "Keep clear of inlet/outlet (rel Nx)", min: 0.0, max: 0.3, step: 0.01, default: 0.12 },
    },
    build: ({ nx, ny, params }) => {
      const b = new MaskBuilder(nx, ny, CELL.FLUID);
      makeDefaultChannel(b, 1.0);

      const localRng = mulberry32(Number(params.seed) >>> 0);
      const count = Math.floor(Number(params.count));
      const rMin = Math.floor(Number(params.rMin));
      const rMax = Math.floor(Number(params.rMax));
      const xPad = Math.floor(nx * Number(params.xPad));

      for (let i = 0; i < count; i++) {
        const r = localRng.int(Math.min(rMin, rMax), Math.max(rMin, rMax));
        const x = localRng.int(1 + xPad, (nx - 2) - xPad);
        const y = localRng.int(1 + r, (ny - 2) - r);
        b.fillCircle(x, y, r, CELL.SOLID);
      }

      return {
        mask: b.mask,
        visType: VisTypes.VORTICITY,
        colormap: VisColormaps.RdBu,
        sim: { inletUx: 0.03, tau: 0.6 }
      };
    },
  },
};
