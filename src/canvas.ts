import { CELL } from "./LBM";

export type RowSpan = { y: number; x0: number; x1: number };
export type PaintCallback = (rows: RowSpan[], value: number) => void;

type Handlers = {
  onDown: (e: PointerEvent) => void;
  onMove: (e: PointerEvent) => void;
  onUpCancel: (e: PointerEvent) => void;
};

export class CanvasPainter {
  #canvas: HTMLCanvasElement;
  #Nx: number;
  #Ny: number;

  #brushRadius = 3; // lattice cells
  #mode: "solid" | "erase" = "solid";
  #isDrawing = false;
  #lastCell: { x: number; y: number } | null = null;
  #onPaint: PaintCallback;

  #handlers?: Handlers;
  #circleCache = new Map<number, Array<{ dy: number; span: number }>>();
  #innerMargin = 2; // keep a small moat near borders by default
  #useRAFCoalescing: boolean;
  #pending: RowSpan[] = [];
  #pendingVal: number | null = null;
  #raf: number = 0;

  constructor(opts: {
    canvas: HTMLCanvasElement;
    Nx: number;
    Ny: number;
    onPaint: PaintCallback; // receives lattice row spans + value
    /**
     * Optional: number of cells kept as an inner margin (unpaintable). Default 2.
     */
    innerMargin?: number;
    /**
     * Optional: if true (default), batches multiple pointer events and emits once per animation frame.
     */
    coalesceWithRAF?: boolean;
  }) {
    this.#canvas = opts.canvas;
    this.#Nx = opts.Nx;
    this.#Ny = opts.Ny;
    this.#onPaint = opts.onPaint;
    if (opts.innerMargin !== undefined)
      this.#innerMargin = Math.max(0, opts.innerMargin | 0);
    this.#useRAFCoalescing = opts.coalesceWithRAF !== false;
  }

  setBrush(radius: number) {
    this.#brushRadius = Math.max(1, radius | 0);
  }
  setModeSolid() {
    this.#mode = "solid";
  }
  setModeErase() {
    this.#mode = "erase";
  }

  /**
   * Set the unpaintable safety margin from borders (in lattice cells).
   * Use 0 to allow painting up to the edge.
   */
  setBoundsMargin(margin: number) {
    this.#innerMargin = Math.max(0, margin | 0);
  }

  enable() {
    if (this.#handlers) return; // already enabled
    const onDown = this.#onPointerDown.bind(this);
    const onMove = this.#onPointerMove.bind(this);
    const onUpCancel = this.#onPointerUpCancel.bind(this);

    this.#canvas.addEventListener("pointerdown", onDown);
    this.#canvas.addEventListener("pointermove", onMove);
    this.#canvas.addEventListener("pointerup", onUpCancel);
    this.#canvas.addEventListener("pointercancel", onUpCancel);

    this.#handlers = { onDown, onMove, onUpCancel };
  }

  disable() {
    if (!this.#handlers) return;
    const { onDown, onMove, onUpCancel } = this.#handlers;
    this.#canvas.removeEventListener("pointerdown", onDown);
    this.#canvas.removeEventListener("pointermove", onMove);
    this.#canvas.removeEventListener("pointerup", onUpCancel);
    this.#canvas.removeEventListener("pointercancel", onUpCancel);
    this.#handlers = undefined;
    this.#flush(true);
  }

  // --- pointer handlers ---

  #onPointerDown(e: PointerEvent) {
    this.#canvas.setPointerCapture?.(e.pointerId);
    this.#isDrawing = true;

    const { x, y } = this.#canvasToCell(e);
    if (!this.#withinInnerBounds(x, y)) return;

    const val = this.#mode === "solid" ? CELL.SOLID : CELL.FLUID;
    const dirty = this.#paintCircle(x, y, this.#brushRadius);
    this.#emit(dirty, val);
    this.#lastCell = { x, y };
  }

  #onPointerMove(e: PointerEvent) {
    if (!this.#isDrawing) return;

    const events = (e.getCoalescedEvents?.() ?? [e]) as PointerEvent[];
    let last = this.#lastCell;
    if (!last) {
      const first = this.#canvasToCell(events[0]);
      if (!this.#withinInnerBounds(first.x, first.y)) return;
      last = first;
    }

    const val = this.#mode === "solid" ? CELL.SOLID : CELL.FLUID;
    const aggregated: RowSpan[] = [];

    for (const evt of events) {
      const { x, y } = this.#canvasToCell(evt);
      if (!this.#withinInnerBounds(x, y)) continue;
      // skip if same lattice cell as previous
      if (last.x === x && last.y === y) continue;

      // capsule raster between last and current sample
      const rows = this.#paintCapsule(last.x, last.y, x, y, this.#brushRadius);
      this.#mergeRowSpansInPlace(aggregated, rows);

      last = { x, y };
    }

    if (aggregated.length) this.#emit(aggregated, val);
    this.#lastCell = last ?? this.#lastCell;
  }

  #onPointerUpCancel(e: PointerEvent) {
    try {
      this.#canvas.releasePointerCapture?.(e.pointerId);
    } catch {}
    this.#isDrawing = false;
    this.#lastCell = null;
    this.#flush(true);
  }

  // --- emit/coalescing ---

  #emit(rows: RowSpan[], val: number) {
    if (!this.#useRAFCoalescing) {
      this.#onPaint(rows, val);
      return;
    }
    this.#mergeRowSpansInPlace(this.#pending, rows);
    this.#pendingVal = val;
    if (this.#raf) return;
    this.#raf = requestAnimationFrame(() => {
      this.#raf = 0;
      if (this.#pending.length) {
        const payload = this.#pending;
        const v = this.#pendingVal ?? val;
        this.#pending = [];
        this.#pendingVal = null;
        this.#onPaint(payload, v);
      }
    });
  }

  #flush(immediate = false) {
    if (!this.#useRAFCoalescing) return;
    if (immediate && this.#raf) {
      cancelAnimationFrame(this.#raf);
      this.#raf = 0;
    }
    if (this.#pending.length) {
      const payload = this.#pending;
      const v =
        this.#pendingVal ?? (this.#mode === "solid" ? CELL.SOLID : CELL.FLUID);
      this.#pending = [];
      this.#pendingVal = null;
      this.#onPaint(payload, v);
    }
  }

  // --- helpers ---

  #canvasToCell(e: PointerEvent) {
    const rect = this.#canvas.getBoundingClientRect();
    const xCss = e.clientX - rect.left;
    const yCss = e.clientY - rect.top;

    let x = Math.floor((this.#Nx * xCss) / rect.width);
    let y = Math.floor((this.#Ny * yCss) / rect.height);

    // flip Y because lattice y=0 is bottom but CSS y=0 is top
    y = this.#Ny - 1 - y;

    // clamp
    x = Math.min(this.#Nx - 1, Math.max(0, x));
    y = Math.min(this.#Ny - 1, Math.max(0, y));
    return { x, y };
  }

  #withinInnerBounds(x: number, y: number) {
    const m = this.#innerMargin;
    return x >= m && x < this.#Nx - m && y >= m && y < this.#Ny - m;
  }

  // --- span utilities ---

  /**
   * Merge row spans in-place: unions intervals that overlap/touch on the same row.
   * This keeps the span list compact and avoids duplicates.
   */
  #mergeRowSpansInPlace(into: RowSpan[], rows: RowSpan[]) {
    if (!rows.length) return;
    // Build a map of y -> intervals (merged)
    const map = new Map<number, Array<{ x0: number; x1: number }>>();

    const add = (y: number, x0: number, x1: number) => {
      let arr = map.get(y);
      if (!arr) {
        arr = [];
        map.set(y, arr);
      }
      // insert and merge (arr is tiny in practice, so linear is fine)
      let inserted = false;
      for (let i = 0; i < arr.length; i++) {
        const seg = arr[i];
        // if overlaps or touches, union
        if (!(x1 < seg.x0 - 1 || x0 > seg.x1 + 1)) {
          seg.x0 = Math.min(seg.x0, x0);
          seg.x1 = Math.max(seg.x1, x1);
          // merge forward if needed
          let j = i + 1;
          while (j < arr.length) {
            const nxt = arr[j];
            if (nxt.x0 <= seg.x1 + 1) {
              seg.x1 = Math.max(seg.x1, nxt.x1);
              arr.splice(j, 1);
            } else break;
          }
          inserted = true;
          break;
        } else if (x1 < seg.x0) {
          arr.splice(i, 0, { x0, x1 });
          inserted = true;
          break;
        }
      }
      if (!inserted) arr.push({ x0, x1 });
    };

    // seed with existing "into"
    for (const r of into) add(r.y, r.x0, r.x1);
    // add new rows
    for (const r of rows) add(r.y, r.x0, r.x1);

    // write back compact list
    into.length = 0;
    for (const [y, arr] of map) {
      for (const seg of arr) into.push({ y, x0: seg.x0, x1: seg.x1 });
    }
  }

  // --- circle & capsule rasterization ---

  #getCircleProfile(r: number) {
    let prof = this.#circleCache.get(r);
    if (prof) return prof;
    const r2 = r * r;
    prof = [];
    for (let dy = -r; dy <= r; dy++) {
      const span = Math.floor(Math.sqrt(Math.max(0, r2 - dy * dy)));
      prof.push({ dy, span });
    }
    this.#circleCache.set(r, prof);
    return prof;
  }

  // returns row spans (does NOT write any buffers)
  #paintCircle(cx: number, cy: number, r: number): RowSpan[] {
    const rows: RowSpan[] = [];
    const prof = this.#getCircleProfile(r);
    for (const { dy, span } of prof) {
      const y = cy + dy;
      if (y < 0 || y >= this.#Ny) continue;
      const x0 = Math.max(0, cx - span);
      const x1 = Math.min(this.#Nx - 1, cx + span);
      rows.push({ y, x0, x1 });
    }
    return rows;
  }

  /**
   * Rasterize a thick segment (capsule) between two lattice points with radius r.
   * Produces row spans without per-step circle stamping.
   */
  #paintCapsule(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    r: number
  ): RowSpan[] {
    if (x0 === x1 && y0 === y1) return this.#paintCircle(x0, y0, r);

    const rows: RowSpan[] = [];
    // Ensure y0 <= y1 for iteration
    if (y0 > y1) {
      [x0, x1, y0, y1] = [x1, x0, y1, y0];
    }

    const yStart = Math.max(0, Math.min(y0, y1) - r);
    const yEnd = Math.min(this.#Ny - 1, Math.max(y0, y1) + r);

    for (let y = yStart; y <= yEnd; y++) {
      // conservative x bounds to search within
      let left = Math.max(0, Math.min(x0, x1) - r);
      let right = Math.min(this.#Nx - 1, Math.max(x0, x1) + r);

      // tighten with small integer searches at edges based on distance^2 to segment
      while (
        left <= right &&
        this.#dist2ToSegment(left, y, x0, y0, x1, y1) > r * r
      )
        left++;
      while (
        right >= left &&
        this.#dist2ToSegment(right, y, x0, y0, x1, y1) > r * r
      )
        right--;

      if (left <= right) rows.push({ y, x0: left, x1: right });
    }

    return rows;
  }

  #dist2ToSegment(
    px: number,
    py: number,
    x0: number,
    y0: number,
    x1: number,
    y1: number
  ) {
    const vx = x1 - x0,
      vy = y1 - y0;
    const wx = px - x0,
      wy = py - y0;
    const c1 = vx * wx + vy * wy;
    if (c1 <= 0) return wx * wx + wy * wy;
    const c2 = vx * vx + vy * vy;
    if (c2 <= c1) {
      const dx = px - x1,
        dy = py - y1;
      return dx * dx + dy * dy;
    }
    const b = c1 / c2;
    const bx = x0 + b * vx,
      by = y0 + b * vy;
    const dx = px - bx,
      dy = py - by;
    return dx * dx + dy * dy;
  }
}
