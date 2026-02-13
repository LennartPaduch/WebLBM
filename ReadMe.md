# WebLBM

Fast, memory-efficient Lattice Boltzmann CFD (D2Q9) in the browser via WebGPU.

[Live Demo](https://weblbm.pages.dev/)

## Overview

WebLBM runs a 2D D2Q9 LBM solver fully on the GPU with WGSL compute shaders.
The implementation is optimized for speed and memory efficiency with:

- Shifted DDFs
- FP16 storage (`f16`) with FP32 arithmetic
- Esoteric Pull in-place streaming
- SoA layout for populations

This project is strongly inspired by Moritz Lehmann's work, especially FluidX3D and related LBM optimization papers.

Startup defaults:

- Preset: `Von Kármán Street`
- Resolution: `1024 x 1024`

## Requirements

- Node.js + npm for local development
- A WebGPU-enabled browser
- `shader-f16` GPU feature support (required by the app)

If `shader-f16` is unavailable, initialization fails with an explicit error message.

## Quick Start

```bash
npm install
npm run dev
```

## Runtime Controls

- `Restart Sim`: reinitializes the current simulation state
- `Pause/Resume`: toggles stepping
- `Settings` / `Hide`: toggles the control panel visibility; on small screens settings use a dedicated full-screen view
- `Scene` selector: switches between geometry presets
- `Resolution`: `256`, `512`, `1024`, `2048` (power-of-two options, clamped by hardware limits and app cap)
- `Vis Type`: velocity magnitude or vorticity
- `Colormap`: `VIRIDIS`, `TURBO`, `RdBu`
- `Paint` mode: adds solid cells
- `Erase` mode: restores fluid cells
- `Reset Paint`: clears all manual paint/erase changes and restores the current scene's default cells
- `Brush size` slider: controls paint radius

## Runtime Metrics

- `Total MLUPS`: end-to-end throughput using all cells
- `Fluid MLUPS`: end-to-end throughput excluding solid-mask cells

## Presets

Current built-in scenes:

- `Empty Tunnel`
- `Von Kármán Street`
- `Staggered Grid`
- `Backward-Facing Step`
- `Venturi (Nozzle)`
- `Porous Media (Random Disks)`

## Technical Notes

### Shifted DDFs (Skordos, 1993)

The solver works with shifted populations:

$f_i^{\text{shifted}} = f_i - w_i$

where $w_i = f_i^{\mathrm{eq}}(\rho = 1, \vec{u} = 0)$ are the lattice weights.

Macroscopic fields are recovered from shifted populations as:

- $\rho = 1 + \sum_i f_i^{\text{shifted}}$
- $\vec{u} = \frac{1}{\rho}\sum_i \left(\mathbf{c}_i f_i^{\text{shifted}}\right)$

This is exactly how density/velocity are reconstructed in the step shader.

Paper: <https://arxiv.org/abs/comp-gas/9306002>

### FP16 storage

DDFs are stored as scaled `f16` values and decoded to `f32` for arithmetic.
This reduces memory traffic and capacity pressure for large grids.
With a $2^{15}$ scaling, represented values cover about $[-2, 2]$ in shifted space,
which matches the typical DDF magnitude range well.

```wgsl
const FP16S_SCALE     : f32 = 32768.0;       // 2^15
const FP16S_INV_SCALE : f32 = 1.0 / 32768.0; // 2^-15

fn decode_f16s(p: f16) -> f32 {
  return f32(p) * FP16S_INV_SCALE;
}

fn pack_f16s(v: f32) -> f16 {
  return f16(v * FP16S_SCALE);
}
```

Paper: <https://epub.uni-bayreuth.de/id/eprint/6559/>

### Collision model (BGK / SRT)

Collision uses single-relaxation-time BGK:

$f_i \leftarrow (1 - \omega)f_i + \omega f_i^{\mathrm{eq}}$

with:

- $\omega = \frac{1}{\tau}$
- $\nu = c_s^2\left(\tau - \frac{1}{2}\right)$, with $c_s^2 = \frac{1}{3}$ for D2Q9

The same `omega`/`tau` relation is used in the implementation.

### Esoteric Pull (Lehmann, 2022)

Streaming is done in-place (single-population storage, parity-controlled access)
instead of ping-pong buffers.

Paper: <https://www.mdpi.com/2079-3197/10/6/92>

### Memory layout and mask

Populations are stored in SoA form:

`f[dir * C + cell]`, where `C = Nx * Ny`, `dir in [0..8]`.

The stepping shader uses bitmask-based wrap indexing, so power-of-two grid sizes are enforced in the UI.

Mask flags:

- `CELL.FLUID`: fluid cell
- `CELL.SOLID`: bounce-back solid
- `CELL.EQ`: equilibrium boundary (inlet/outlet)

## Acknowledgments

This project was inspired by Moritz Lehmann's FluidX3D work and related LBM optimization papers.
FluidX3D: <https://github.com/ProjectPhysX/FluidX3D>

## Browser Support

![WebGPU browser support](https://caniuse.bitsofco.de/image/webgpu.png)
