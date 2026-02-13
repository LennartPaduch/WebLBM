struct StepParams {
  Nx:        u32,
  Ny:        u32,
  cellCount: u32,
  Q:         u32,

  rhoIn:     f32,
  uInx:      f32,
  uIny:      f32,
  // omega = 1/tau
  omega:     f32,
};

  // Toggled each step for Esoteric Pull access pattern.
  struct StepDynamic {
    parity: u32,
    _pad0:  u32,
    _pad1:  u32,
    _pad2:  u32,
  };

@group(0) @binding(0) var<storage, read_write> f           : array<f16>; // SoA populations
@group(0) @binding(1) var<uniform>             P           : StepParams;
@group(0) @binding(2) var<storage, read>       mask        : array<u32>;
@group(0) @binding(3) var<uniform>             Pd          : StepDynamic;


fn load_f_ep_implicit(cell:u32, parity:u32, C:u32, j: array<u32, 9>) -> array<f32,9> {
  // In-place Esoteric Pull readback.
  var fi : array<f32,9u>;

  fi[0] = decode_f16s(f[addr(0u, cell, C)]); 
  if (parity == 0u) {
    for (var i=1u; i<9u; i+=2u) {
      fi[i   ] = decode_f16s(f[addr(i, cell, C)]);
      fi[i+1u] = decode_f16s(f[addr(i+1u, j[i], C)]);
    }
  } else {
    for (var i=1u; i<9u; i+=2u) {
      fi[i   ] = decode_f16s(f[addr(i+1u, cell, C)]);
      fi[i+1u] = decode_f16s(f[addr(i, j[i], C)]);
    }
  }

  return fi;
}

fn macros_from_shifted_d2q9(fi: ptr<function, array<f32, 9>>, rho_out: ptr<function, f32>, ux_out: ptr<function, f32>, uy_out: ptr<function, f32>) {
  let f0 = (*fi)[0];
  let f1 = (*fi)[1];
  let f2 = (*fi)[2];
  let f3 = (*fi)[3];
  let f4 = (*fi)[4];
  let f5 = (*fi)[5];
  let f6 = (*fi)[6];
  let f7 = (*fi)[7];
  let f8 = (*fi)[8];

  var rho = f0;
  rho += f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
  rho += 1.0;

  let axis_x = f1 - f2;
  let axis_y = f3 - f4;
  let diag_a = f5 - f6;
  let diag_b = f7 - f8;

  let mx = axis_x + diag_a + diag_b;
  let my = axis_y + diag_a - diag_b;
  let inv_rho = 1.0 / rho;

  *rho_out = rho;
  *ux_out = mx * inv_rho;
  *uy_out = my * inv_rho;
}

@compute @workgroup_size(WGX, WGY, WGZ)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.Nx || gid.y >= P.Ny) { return; }

  let cell : u32 = gid.x + gid.y * P.Nx;
  let C    : u32 = P.cellCount;

  let m = mask[cell];
  if (is_solid(m)) { return; }

  let j  = get_neighbors(gid.x,gid.y);
  var fi = load_f_ep_implicit(cell, Pd.parity, C, j);

  var rhon: f32;
  var uxn : f32;
  var uyn : f32;

  if (is_eq(m)) {
    if (gid.x == 0u) {
      // Left inlet.
      rhon = P.rhoIn;
      uxn  = P.uInx;
      uyn  = P.uIny;
    } else if (gid.x == P.Nx-1u) {
      // Right outlet from inner-cell macros.
      let innerX = P.Nx - 2u;
      let innerCell = innerX + gid.y * P.Nx;
      let innerJ = get_neighbors(innerX, gid.y);
      var innerFi = load_f_ep_implicit(innerCell, Pd.parity, C, innerJ);
      macros_from_shifted_d2q9(&innerFi, &rhon, &uxn, &uyn);
    } else {
      macros_from_shifted_d2q9(&fi, &rhon, &uxn, &uyn);
    }
  } else {
    macros_from_shifted_d2q9(&fi, &rhon, &uxn, &uyn);
  }

  let feq = feq_d2q9_shifted(rhon, vec2<f32>(uxn, uyn));

  let one_minus_omega = 1.0 - P.omega;
  if (is_eq(m)) {
    for (var i=0u; i<9u; i++) {
      fi[i] = feq[i];
    }
  } else {
    for (var i=0u; i<9u; i++){
      fi[i] = fma(P.omega, feq[i], one_minus_omega * fi[i]);
    }
  }

  f[addr(0u, cell, C)] = pack_f16s(fi[0]);
  if (Pd.parity == 0u) {
    for (var i=1u; i<9u; i+=2u) {
      f[addr(i+1u, j[i], C)] = pack_f16s(fi[i   ]);
      f[addr(i,    cell, C)] = pack_f16s(fi[i+1u]);
    }
  } else {
    for (var i=1u; i<9u; i+=2u) {
      f[addr(i,    j[i], C)] = pack_f16s(fi[i   ]);
      f[addr(i+1u, cell, C)] = pack_f16s(fi[i+1u]);
    }
  }
}
