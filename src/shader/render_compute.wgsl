// -----------------------------------------------------------------------------
// LBM Heatmap Render (WGSL) — EP-consistent load, Viridis/Turbo colormaps
// P.mode: 0=rho, 1=|u|
// P.cmap: 0=Viridis, 1=Turbo
// P.vmin/vmax: linear range for normalization
// -----------------------------------------------------------------------------

struct VisParams {
  Nx:        u32,
  Ny:        u32,
  cellCount: u32,
  mode:      u32, // 0=rho, 1=|u|
  cmap:      u32, // 0=Viridis, 1=Turbo
  vmin:      f32,
  vmax:      f32
};

// Dynamic params: updated every step ----
struct StepDynamic {
  parity: u32,
  _pad:   vec3<u32> // pad to 16 bytes to be uniform-safe
};


// --- bindings ----------------------------------------------------------------
@group(0) @binding(0) var<storage, read>        f    : array<f16>;  // SoA: f[dir*C + cell]
@group(0) @binding(1) var<uniform>              P    : VisParams;
@group(0) @binding(2) var<uniform>              Pd   : StepDynamic;
@group(0) @binding(3) var<storage, read>        mask : array<u32>;
@group(0) @binding(4) var outputTex : texture_storage_2d<rgba8unorm, write>;


fn load_f_ep_implicit(cell:u32, parity:u32, C:u32, Nx:u32, Ny:u32, j: array<u32, 9>) -> array<f32,9> {
  var fi : array<f32,9>;

  fi[0] = decode_f16s(f[addr(0u, cell, C)]); 
  for(var i=1u; i<9u; i+=2u){
    fi[i   ] = decode_f16s(f[addr(select(i   , i+1u, parity == 1u), cell, C)]);
		fi[i+1u] = decode_f16s(f[addr(select(i+1u, i   , parity == 1u), j[i], C)]);
  }

  return fi;
}

// ---- normalization ------------------------------------------------------------
fn normalize01(s:f32, vmin:f32, vmax:f32) -> f32 {
  let eps = 1e-12;
  let d = max(vmax - vmin, eps);
  return clamp((s - vmin) / d, 0.0, 1.0);
}

// ---- small LUT samplers for professional colormaps ---------------------------

const VIRIDIS_LUT : array<vec3<f32>, 10> = array<vec3<f32>,10>(
  vec3<f32>(0.267004, 0.004874, 0.329415),
  vec3<f32>(0.282327, 0.094955, 0.417331),
  vec3<f32>(0.253935, 0.265254, 0.529983),
  vec3<f32>(0.206756, 0.371758, 0.553117),
  vec3<f32>(0.163625, 0.471133, 0.558148),
  vec3<f32>(0.128729, 0.567573, 0.551864),
  vec3<f32>(0.134692, 0.658636, 0.517649),
  vec3<f32>(0.266941, 0.748751, 0.440573),
  vec3<f32>(0.477504, 0.821444, 0.318195),
  vec3<f32>(0.741388, 0.873449, 0.149561)
);

const TURBO_LUT : array<vec3<f32>, 10> = array<vec3<f32>,10>(
  vec3<f32>(0.18995, 0.07176, 0.23217),
  vec3<f32>(0.25107, 0.25237, 0.63374),
  vec3<f32>(0.27628, 0.51592, 0.85877),
  vec3<f32>(0.23389, 0.70494, 0.69883),
  vec3<f32>(0.15338, 0.80480, 0.49659),
  vec3<f32>(0.21230, 0.83660, 0.27766),
  vec3<f32>(0.48224, 0.80874, 0.11465),
  vec3<f32>(0.80462, 0.64781, 0.04719),
  vec3<f32>(0.98360, 0.39654, 0.13090),
  vec3<f32>(0.98730, 0.17860, 0.16498)
);

// Smoothstep mix between LUT entries
fn sample_lut10(t:f32, lut: ptr<function, array<vec3<f32>,10>>) -> vec3<f32> {
  let x = clamp(t, 0.0, 1.0) * 9.0;
  let i = u32(floor(x));
  let j = min(i + 1u, 9u);
  let u = smoothstep(0.0, 1.0, fract(x));
  return mix((*lut)[i], (*lut)[j], u);
}

fn colormapViridis(t:f32) -> vec3<f32> {
  var lut = VIRIDIS_LUT;
  return sample_lut10(t, &lut);
}
fn colormapTurbo(t:f32) -> vec3<f32> {
  var lut = TURBO_LUT;
  return sample_lut10(t, &lut);
}

@compute @workgroup_size(WGX, WGY, WGZ)
fn render(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.Nx || gid.y >= P.Ny) { return; }
  let cell : u32 = gid.x + gid.y * P.Nx;

  if (mask[cell] == CELL_SOLID) {
    textureStore(outputTex, vec2<i32>(i32(gid.x), i32(gid.y)),
                 vec4<f32>(0.10, 0.10, 0.10, 1.0)); // dark gray solids
    return;
  } else if (mask[cell] == CELL_EQ) {
    textureStore(outputTex, vec2<i32>(i32(gid.x), i32(gid.y)),
                 vec4<f32>(1.0, 0.0, 0.0, 1.0)); // red
    return;
  }

  let j  = get_neighbors(cell);
  // EP-consistent populations at this cell
  let fi = load_f_ep_implicit(cell, Pd.parity, P.cellCount, P.Nx, P.Ny, j);

  // Macros (shifted DDFs): rho starts at 1.0
  var rho : f32 = 1.0;
  var ux  : f32 = 0.0;
  var uy  : f32 = 0.0;
  for (var d:u32 = 0u; d < 9u; d++) {
    let v = fi[d];
    rho += v;
    ux  += v * f32(EX[d]);
    uy  += v * f32(EY[d]);
  }
  ux /= rho; uy /= rho;

  // Choose scalar (fixed: mode 0=rho, 1=|u|)
  let speed = sqrt(ux*ux + uy*uy);
  let s = select(speed, rho, P.mode == 1u);

  // Normalize & colorize
  let t = normalize01(s, P.vmin, P.vmax);
  let rgb = select(colormapViridis(t), colormapTurbo(t), P.cmap == 1u);

  textureStore(outputTex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(rgb, 1.0));
}
