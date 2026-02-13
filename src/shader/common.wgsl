  enable f16;

  const CELL_FLUID  : u32 = 0u;
  const CELL_SOLID  : u32 = 1u << 0;
  const CELL_EQ     : u32 = 1u << 1;

  override WGX: u32 = 64u;
  override WGY: u32 = 4u; 
  override WGZ: u32 = 1u;

  const EX  : array<i32,9> = array<i32,9>(0, 1,-1, 0, 0, 1,-1, 1,-1); // C, E, W, N, S, NE, SW, SE, NW
  const EY  : array<i32,9> = array<i32,9>(0, 0, 0, 1,-1, 1,-1,-1, 1);
  const OPP : array<u32,9> = array<u32,9>(
    0u,
    2u,
    1u,
    4u,
    3u,
    6u,
    5u,
    8u,
    7u
  );

  const W0 : f32 = 4.0 / 9.0;
  const WS : f32 = 1.0 / 9.0;
  const WE : f32 = 1.0 / 36.0;

  const Q  : u32 = 9u;

  const FP16S_SCALE      : f32 = 32768.0;
  const FP16S_INV_SCALE  : f32 = 1.0 / 32768.0;

  fn decode_f16s(p: f16) -> f32 {
    return f32(p) * FP16S_INV_SCALE;
  }

  fn pack_f16s(v: f32) -> f16 {
    return f16(v * FP16S_SCALE);
  }

  fn is_fluid(m:u32) -> bool { return (m == 0);}
  fn is_solid(m:u32) -> bool { return (m & CELL_SOLID)  != 0u; }
  fn is_eq(m:u32)    -> bool { return (m & CELL_EQ)     != 0u; } // equilibrium boundary marker

  // SoA layout: f[dir * C + cell]
  fn addr(dir:u32, cell:u32, C:u32) -> u32 { return dir*C + cell; }

  fn calculate_indices_xy(
    x: u32,
    y: u32,
    x0: ptr<function, u32>,
    xp: ptr<function, u32>,
    xm: ptr<function, u32>,
    y0: ptr<function, u32>,
    yp: ptr<function, u32>,
    ym: ptr<function, u32>
  ) {
    *x0 = x;

    // Bitmask wrap requires power-of-two dimensions.
    let maskX = P.Nx - 1u;
    let maskY = P.Ny - 1u;

    *xp = (x + 1u) & maskX;
    *xm = (x - 1u) & maskX;

    *y0 = y * P.Nx;
    *yp = ((y + 1u) & maskY) * P.Nx;
    *ym = ((y - 1u) & maskY) * P.Nx;
  }

  fn get_neighbors(x: u32, y: u32) -> array<u32, 9>{
    var j: array<u32, 9>;

    var x0: u32;
    var xp: u32;
    var xm: u32;
    var y0: u32;
    var yp: u32;
    var ym: u32;

    calculate_indices_xy(x, y, &x0, &xp, &xm, &y0, &yp, &ym);
    
    j[1] = xp + y0;
    j[2] = xm + y0;
    j[3] = x0 + yp;
    j[4] = x0 + ym;

    j[5] = xp + yp;
    j[6] = xm + ym;
    j[7] = xp + ym;
    j[8] = xm + yp;

    return j;
  }

  fn feq_d2q9_shifted(rho_in: f32, u_in: vec2<f32>) -> array<f32, 9> {
    var out : array<f32, 9>;
    let rho : f32 = rho_in;
    let rho_shift : f32 = rho - 1.0;

    let ux3: f32 = 3.0 * u_in.x;
    let uy3: f32 = 3.0 * u_in.y;
    let c3 : f32 = -3.0 * dot(u_in, u_in);

    let rho_s   : f32 = WS * rho;
    let rho_e   : f32 = WE * rho;
    let shift_s : f32 = WS * rho_shift;
    let shift_e : f32 = WE * rho_shift;

    out[0] = W0 * fma(rho, 0.5 * c3, rho_shift);

    let ux_term : f32 = ux3 * ux3 + c3;
    let uy_term : f32 = uy3 * uy3 + c3;
    out[1] = rho_s * (0.5 * ux_term + ux3) + shift_s;
    out[2] = rho_s * (0.5 * ux_term - ux3) + shift_s;
    out[3] = rho_s * (0.5 * uy_term + uy3) + shift_s;
    out[4] = rho_s * (0.5 * uy_term - uy3) + shift_s;

    let up      : f32 = ux3 + uy3;
    let um      : f32 = ux3 - uy3;
    let up_term : f32 = up * up + c3;
    let um_term : f32 = um * um + c3;
    out[5] = rho_e * (0.5 * up_term + up) + shift_e;
    out[6] = rho_e * (0.5 * up_term - up) + shift_e;
    out[7] = rho_e * (0.5 * um_term + um) + shift_e;
    out[8] = rho_e * (0.5 * um_term - um) + shift_e;

    return out;
  }
