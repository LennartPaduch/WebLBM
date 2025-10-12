enable f16;

const CELL_FLUID  : u32 = 0u;
const CELL_SOLID  : u32 = 1u << 0;
const CELL_EQ     : u32 = 1u << 1;  // TYPE_E (equilibrium BC: inlet/outlet)

const WGX = 32u;
const WGY = 32u; 
const WGZ =  1u;

const EX  : array<i32,9> = array<i32,9>(0, 1,-1, 0, 0, 1,-1, 1,-1); // C, E, W, N, S, NE, SW, SE, NW
const EY  : array<i32,9> = array<i32,9>(0, 0, 0, 1,-1, 1,-1,-1, 1);
const OPP : array<u32,9> = array<u32,9>(
  0u, // C <-> C
  2u, // E <-> W
  1u, // W <-> E
  4u, // N <-> S
  3u, // S <-> N
  6u, // NE <-> SW
  5u, // SW <-> NE
  8u, // SE <-> NW
  7u  // NW <-> SE
);

// Opposite pairs
const PAIRS = array<vec2<u32>,4>(
  vec2<u32>(1u, 2u), // E <-> W
  vec2<u32>(3u, 4u), // N <-> S
  vec2<u32>(5u, 6u), // NE <-> SW
  vec2<u32>(7u, 8u)  // SE <-> NW
);

const W0 : f32 = 4.0 / 9.0;
const WS : f32 = 1.0 / 9.0;
const WE : f32 = 1.0 / 36.0;

const Q  : u32 = 9u;

const FP16S_SCALE      : f32 = 32768.0;          // 2^15
const FP16S_INV_SCALE  : f32 = 1.0 / 32768.0;    // 2^-15

fn decode_f16s(p: f16) -> f32 {
  return f32(p) * FP16S_INV_SCALE;          // unpack + downscale
}

fn store_f16s(p: ptr<storage, f16, read_write>, v: f32) {
  *p = f16(v * FP16S_SCALE);                 // upscale + pack (round-to-nearest-even)
}

fn is_fluid(m:u32)  -> bool { return (m == 0);}
fn is_solid(m:u32)  -> bool { return (m & CELL_SOLID)  != 0u; }
fn is_eq(m:u32)     -> bool { return (m & CELL_EQ)     != 0u; } // TYPE_E (equilibrium BC: inlet/outlet)

// Addressing (SoA): f[dir*C + cell]
fn addr(dir:u32, cell:u32, C:u32) -> u32 { return dir*C + cell; }

fn coordinates(cell: u32, Nx:u32) -> vec2<u32> {
  let x = cell % Nx;
  let y = cell / Nx;
  return vec2<u32>(x, y);
}

// In-bounds check (non-periodic)
fn in_bounds(ix:i32, iy:i32, Nx:u32, Ny:u32) -> bool {
  return (ix >= 0 && iy >= 0 && ix < i32(Nx) && iy < i32(Ny));
}

fn calculate_indices(
  cell: u32,
  x0:   ptr<function, u32>, 
  xp:   ptr<function, u32>, 
  xm:   ptr<function, u32>, 
  y0:   ptr<function, u32>, 
  yp:   ptr<function, u32>, 
  ym:   ptr<function, u32>){

  let xy: vec2<u32> = coordinates(cell, P.Nx);
  *x0 = xy.x;
  *xp = (xy.x + 1u) % P.Nx;
  *xm = (xy.x + P.Nx - 1u) % P.Nx;
  *y0 = xy.y * P.Nx;
  *yp = ((xy.y + 1u) % P.Ny) * P.Nx;
  *ym = ((xy.y + P.Ny - 1u) % P.Ny) * P.Nx;
}

fn get_neighbors(cell: u32) -> array<u32, 9>{
  var j: array<u32, 9>;

  var x0: u32;
  var xp: u32;
  var xm: u32;
  var y0: u32;
  var yp: u32;
  var ym: u32;

  calculate_indices(cell, &x0, &xp, &xm, &y0, &yp, &ym);

  j[1] = xp+y0; j[2] = xm+y0; // +00 -00
  j[3] = x0+yp; j[4] = x0+ym; // 0+0 0-0
  j[5] = xp+yp; j[6] = xm+ym; // ++0 --0
  j[7] = xp+ym; j[8] = xm+yp; // +-0 -+0

  return j;
}

// Access u as two stacked planes: [0..C) = ux, [C..2C) = uy
fn feq_d2q9_shifted(rho_in: f32, u_in: vec2<f32>) -> array<f32, 9> {
  var out : array<f32, 9>;
  // local copies
  var ux : f32 = u_in.x;
  var uy : f32 = u_in.y;
  let rho   : f32 = rho_in;
  let rhom1 : f32 = rho - 1.0; // rhom1 is arithmetic optimization to minimize digit extinction

  // c3 = -3 * (ux^2 + uy^2)
  let c3 : f32 = -3.0 * (ux*ux + uy*uy);

  // scale velocities by 3 (1/cs^2 with cs^2=1/3)
  ux = ux * 3.0;
  uy = uy * 3.0;

  // weights * rho and weights * (rho-1)
  let rhos   : f32 = WS * rho;
  let rhoe   : f32 = WE * rho;
  let rhom1s : f32 = WS * rhom1;
  let rhom1e : f32 = WE * rhom1;

  // center (shifted)
  out[0] = W0 * (rho * (0.5 * c3) + rhom1);

  // precombinations
  let u_plus  : f32 = ux + uy; // dot for NE/SE pairs
  let u_minus : f32 = ux - uy; // dot for NW/SE pairs

  // cardinals, mapped to EX/EY:
  out[1] = rhos * (0.5 * (ux*ux + c3) + ux) + rhom1s; // E  ( +x)
  out[2] = rhos * (0.5 * (ux*ux + c3) - ux) + rhom1s; // W  ( -x)
  out[3] = rhos * (0.5 * (uy*uy + c3) + uy) + rhom1s; // N  ( +y)
  out[4] = rhos * (0.5 * (uy*uy + c3) - uy) + rhom1s; // S  ( -y)

  out[5] = rhoe * (0.5 * (u_plus*u_plus  + c3) +  u_plus ) + rhom1e; // NE
  out[8] = rhoe * (0.5 * (u_minus*u_minus + c3) -  u_minus) + rhom1e; // NW
  out[6] = rhoe * (0.5 * (u_plus*u_plus  + c3) -  u_plus ) + rhom1e; // SW
  out[7] = rhoe * (0.5 * (u_minus*u_minus + c3) +  u_minus) + rhom1e; // SE

  return out;
}

