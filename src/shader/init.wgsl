struct Params {
  Nx:u32,
  Ny:u32, 
  Q:u32,
};

@group(0) @binding(0) var<storage, read_write> f    : array<f16>;   // current
@group(0) @binding(1) var<storage, read>       mask : array<u32>;
@group(0) @binding(2) var<uniform>             P    : Params;
@group(0) @binding(3) var<storage, read_write> global_u    : array<f32>;   // 2*C length: ux, uy
@group(0) @binding(4) var<storage, read_write> global_rho  : array<f32>;

fn ux_at(cell:u32, C:u32) -> f32 { return global_u[cell]; }
fn uy_at(cell:u32, C:u32) -> f32 { return global_u[C + cell]; }

@compute @workgroup_size(WGX,WGY, WGZ)
fn initialize(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.Nx || gid.y >= P.Ny) { return; }
  let C    : u32 = P.Nx * P.Ny;
  let cell : u32 = gid.x + gid.y * P.Nx;

  let m = mask[cell];

  var r  : f32 = 1.0; 
  var ux : f32 = 0.0; 
  var uy : f32 = 0.0; 

  if(is_eq(m)){
    if(gid.x == P.Nx -1){
      global_rho[cell] = 1.0;
      global_u[cell] = 0.0;
      global_u[C+cell] = 0.0;
      let feq = feq_d2q9_shifted(global_rho[cell], vec2<f32>(global_u[cell], global_u[C + cell]));
      for (var d:u32 = 0u; d < 9u; d++) { store_f16s(&f[addr(d, cell, C)], feq[d]); }
      return;
    }
    global_rho[cell] = 1.0;
    global_u[cell] = 0.05;
    global_u[C+cell] = 0.0;
    let feq = feq_d2q9_shifted(global_rho[cell], vec2<f32>(global_u[cell], global_u[C + cell]));
    for (var d:u32 = 0u; d < 9u; d++) { store_f16s(&f[addr(d, cell, C)], feq[d]); }
    return;
  }

  if (!is_eq(m) && !is_solid(m)) {
    global_rho[cell]      = r;
    global_u[cell]        = ux;
    global_u[C + cell]    = uy;
  }

  if (is_solid(m)) {
    global_rho[cell] = 1.0;
    global_u[cell] = 0.0;
    global_u[C+cell] = 0.0;
    r  = 1.0;
    ux = 0.0;
    uy = 0.0;
  }

  // Build equilibrium with shifted DDFs (Skordos, 1993) (ρ starts at ~1.0)
  let feq = feq_d2q9_shifted(r, vec2<f32>(ux, uy));

  // Write equilibrium into f 
  for (var d:u32 = 0u; d < 9u; d++) {
    store_f16s(&f[addr(d, cell, C)],feq[d]);
  }
}

