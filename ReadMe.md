# WebLBM

Fast and memory efficient Lattice Boltzmann CFD (D2Q9) for the browser running on the GPU via the WebGPU API.
<br>
[Live Demo](https://weblbm.pages.dev/)

## Overview

This repository implements the Lattice Boltzmann Method (LBM) for the two-dimensional case (D2Q9) running on the GPU using the WebGPU API and its shading language (.wgsl). It leverages WebGPU for high-performance, parallel computations directly in the browser, making CFD simulations accessible without specialized hardware or software.
The implementation is highly optimized for speed, precision, and memory efficiency, incorporating advanced techniques like shifted density distribution functions (DDFs), FP16s storage, and the Esoteric Pull scheme.

## Technical Details

### Shifted DDFs (Skordos, 1993):

To achieve maximum accuracy, the implementation works with shifted DDFs: $f_i^{\text{shifted}} := f_i - w_i$
where $w_i = f_i^{\mathrm{eq}}(\rho = 1, \vec{u} = 0)$ are the lattice weights with $\rho$ and $\vec{u}$ being the local fluid density and
velocity.
<br>[Link to Paper](https://arxiv.org/abs/comp-gas/9306002)

### FP16s

DDFs are stored in FP16 format instead of FP32 or FP64, saving 50% in memory compared to FP32 or 75% compared to FP64. This enables larger lattice grids, as LBM is memory-bound. Accuracy is not compromised, since FP32/FP64 often underutilize bits for typical DDF ranges, and we enhance FP16 precision by shifting values by $2^{-15}$ for an effective range of ±2 (where DDFs typically stay within ±1).
<br>[Link to Paper](https://epub.uni-bayreuth.de/id/eprint/6559/)

Arithmetic is performed in FP32 for stability, with encoding/decoding as follows:

```wgsl
const FP16S_SCALE      : f32 = 32768.0;          // 2^15
const FP16S_INV_SCALE  : f32 = 1.0 / 32768.0;    // 2^-15

fn decode_f16s(p: f16) -> f32 {
  return f32(p) * FP16S_INV_SCALE; // unpack + downscale
}

fn store_f16s(p: ptr<storage, f16, read_write>, v: f32) {
  *p = f16(v * FP16S_SCALE); // upscale + pack
}
```

### Esoteric Pull (Lehmann, 2022)

Compute shaders run in parallel and arbitrary order, requiring either data independence or a ping-pong buffer scheme. The Esoteric Pull is an in-place method that allows using a single copy of DDFs, reducing memory usage while maintaining correctness.
<br>[Link to Paper](https://www.mdpi.com/2079-3197/10/6/92)

### Memory Layout & Data Model

Structure-of-Arrays (SoA)
All populations live in a single SoA buffer:

```
f[dir*C + cell] //C = Nx*Ny, dir ∈ [0..8]
```

This improves coalescing on the GPU when reading a fixed direction for many cells.

### Domain Mask

An integer mask per cell encodes material/BC type:

```
CELL_FLUID: Fluid cells
CELL_SOLID: bounce-back via Esoteric Pull (implicit, in-place)
CELL_EQ   : equilibrium boundary (used for inlet/outlet)
```

### Browser Support

![WebGPU browser support](https://caniuse.bitsofco.de/image/webgpu.png)
