#include common;

struct StepParams {
  Nx:        u32,
  Ny:        u32,
  cellCount: u32,
  Q:         u32,

  rhoIn:     f32,
  uInx:      f32,
  uIny:      f32,
  rhoOut:    f32,

  omega:     f32, // LBM relaxation rate w = dt/tau = dt/(nu/c^2+dt/2) = 1/(3*nu+1/2)
  _pad0:     u32,
  _pad1:     u32,
  _pad2:     u32,
};

// Dynamic params: updated every step
struct StepDynamic {
  parity: u32,
  _pad:   vec3<u32> // pad to 16 bytes to be uniform-safe
};

@group(0) @binding(0) var<storage, read_write> f           : array<f16>;   // SoA: f[i*C + cell]
@group(0) @binding(1) var<uniform>             P           : StepParams;
@group(0) @binding(2) var<storage, read>       mask        : array<u32>;
@group(0) @binding(3) var<storage, read_write> global_u    : array<f16>;   // 2*C length: ux, uy
@group(0) @binding(4) var<storage, read_write> global_rho  : array<f16>;
@group(0) @binding(5) var<uniform>             Pd          : StepDynamic;


// Esoteric Pull: implicit BB
fn load_f_ep_implicit(cell:u32, parity:u32, C:u32, Nx:u32, Ny:u32, j: array<u32, 9>) -> array<f32,9> {
  var fi : array<f32,9>;

  fi[0] = decode_f16s(f[addr(0u, cell, C)]); 
  for(var i=1u; i<P.Q; i+=2u){
    fi[i   ] = decode_f16s(f[addr(select(i   , i+1u, parity == 1u), cell, C)]);
		fi[i+1u] = decode_f16s(f[addr(select(i+1u, i   , parity == 1u), j[i], C)]);
  }

  return fi;
}

fn calculate_rho_u(fi: ptr<function, array<f32, 9>>, rhon: ptr<function, f32>, uxn: ptr<function, f32>, uyn: ptr<function, f32>) {
    var rho: f32 = (*fi)[0];
    for (var d: u32 = 1u; d < 9u; d++) {  // calculate density from f. 9 for D2Q9, that's just the length of the velocity_set
        rho += (*fi)[d];
    }
    rho += 1.0;  // add 1.0 last to avoid digit extinction effects when summing up f (perturbation method / DDF-shifting)

    var ux: f32 = (*fi)[1] - (*fi)[2] + (*fi)[5] - (*fi)[6] + (*fi)[7] - (*fi)[8]; // calculate velocity from fi (alternating + and - for best accuracy)
    var uy: f32 = (*fi)[3] - (*fi)[4] + (*fi)[5] - (*fi)[6] + (*fi)[8] - (*fi)[7];

    *rhon = rho;
    *uxn  = ux / rho;
    *uyn  = uy / rho;
}

@compute @workgroup_size(WGX, WGY, WGZ)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.Nx || gid.y >= P.Ny) { return; }

  let cell : u32 = gid.x + gid.y * P.Nx;
  let C    : u32 = P.cellCount;

  let m = mask[cell];
  if (is_solid(m)) { return; }

  // --- Load (Esoteric Pull): parity-controlled self/neighbor indices
  let j  = get_neighbors(cell);
  var fi = load_f_ep_implicit(cell, Pd.parity, C, P.Nx, P.Ny, j);

  // --- Collision inputs
  var rhon: f32;
  var uxn : f32;
  var uyn : f32;

  if (gid.x == P.Nx-1u && is_eq(m)) { // outlet
    let inner = (P.Nx-2u) + gid.y*P.Nx;
    global_rho[cell] = global_rho[inner];
    global_u[  cell] = global_u[  inner];
    global_u[C+cell] = global_u[C+inner];
  }

  if (is_eq(m)) { //equilibrium BC: inlet/outlet
    rhon = decode_f16s(global_rho[cell]);
    uxn  = decode_f16s(global_u[  cell]);
    uyn  = decode_f16s(global_u[C+cell]);
  } else {
    calculate_rho_u(&fi, &rhon, &uxn, &uyn); // calculate density and velocity fields from fi
    store_f16s(&global_rho[cell], rhon);
    store_f16s(&global_u[  cell], uxn);
    store_f16s(&global_u[C+cell], uyn);
  }

  // Equilibrium (shifted DDFs)
  let feq = feq_d2q9_shifted(rhon, vec2<f32>(uxn, uyn));

  var Fin: array<f32, 9>;
  for (var i=0u; i<9u; i+=1u) { Fin[i] = 0.0; }

  // SRT (Single-Relaxation-Time (BGK))
  let one_minus_omega = 1.0 - P.omega;
  for (var i=0u; i<9u; i++){
    fi[i] = select(fma(P.omega, feq[i], fma(one_minus_omega, fi[i], Fin[i])), feq[i], is_eq(m)); // perform collision (SRT)
  }

  store_f16s(&f[addr(0u, cell, C)], fi[0]);
  for (var i=1u; i<9u; i+=2u){
    store_f16s(&f[addr(select(i+1u, i   ,  Pd.parity == 1u), j[i], C)], fi[i   ]);
    store_f16s(&f[addr(select(i   , i+1u,  Pd.parity == 1u), cell, C)], fi[i+1u]);
  }  
}
