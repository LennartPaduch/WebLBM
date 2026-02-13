  struct VisParams {
    Nx:        u32,
    Ny:        u32,
    cellCount: u32,
    mode:      u32, // 0=|u|, 1=vorticity
    cmap:      u32, // 0=Viridis, 1=Turbo, 2=RdBu
    vmin:      f32,
    vmax:      f32,
    inletUx:   f32,
    inletUy:   f32,
  };

  struct StepDynamic {
    parity: u32,
    _pad0:  u32,
    _pad1:  u32,
    _pad2:  u32,
  };

  @group(0) @binding(0) var<storage, read>  f          : array<f16>;   // SoA: f[i*C + cell]
  @group(0) @binding(1) var<storage, read>  mask       : array<u32>;
  @group(0) @binding(2) var<uniform>        P          : VisParams;
  @group(0) @binding(3) var<uniform>        Pd         : StepDynamic;
  @group(0) @binding(4) var outputTex : texture_storage_2d<rgba8unorm, write>;

  // ---- Colormaps ----
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

  const RDBU_LUT : array<vec3<f32>, 9> = array<vec3<f32>, 9>(
    vec3<f32>(0.192, 0.212, 0.584),
    vec3<f32>(0.263, 0.447, 0.702),
    vec3<f32>(0.455, 0.678, 0.819),
    vec3<f32>(0.819, 0.898, 0.941),
    vec3<f32>(0.961, 0.961, 0.961),
    vec3<f32>(0.992, 0.858, 0.780),
    vec3<f32>(0.956, 0.647, 0.509),
    vec3<f32>(0.839, 0.376, 0.302),
    vec3<f32>(0.698, 0.094, 0.168)
  );

  fn colormapRdBu(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0) * 8.0;
    let i = u32(floor(x));
    let j = min(i + 1u, 8u);
    let u = smoothstep(0.0, 1.0, fract(x));
    return mix(RDBU_LUT[i], RDBU_LUT[j], u);
  }

  fn colormapViridis(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0) * 9.0;
    let i = u32(floor(x));
    let j = min(i + 1u, 9u);
    let u = smoothstep(0.0, 1.0, fract(x));
    return mix(VIRIDIS_LUT[i], VIRIDIS_LUT[j], u);
  }
  fn colormapTurbo(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0) * 9.0;
    let i = u32(floor(x));
    let j = min(i + 1u, 9u);
    let u = smoothstep(0.0, 1.0, fract(x));
    return mix(TURBO_LUT[i], TURBO_LUT[j], u);
  }

  fn load_f_ep_implicit(cell:u32, parity:u32, C:u32, j: array<u32, 9>) -> array<f32,9> {
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

  fn velocity_from_f(cell: u32, x: u32, y: u32, parity: u32) -> vec2<f32> {
    let j = get_neighbors(x, y);
    var fi = load_f_ep_implicit(cell, parity, P.cellCount, j);

    var rho: f32;
    var ux: f32;
    var uy: f32;
    macros_from_shifted_d2q9(&fi, &rho, &ux, &uy);
    return vec2<f32>(ux, uy);
  }

  fn sample_velocity(cell: u32, x: u32, y: u32, parity: u32) -> vec2<f32> {
    let m = mask[cell];
    if (is_solid(m)) { return vec2<f32>(0.0); }

    if (is_eq(m)) {
      if (x == 0u) {
        return vec2<f32>(P.inletUx, P.inletUy);
      }
      if (x == P.Nx - 1u) {
        var innerX: u32 = 0u;
        if (P.Nx > 1u) { innerX = P.Nx - 2u; }
        let innerCell = innerX + y * P.Nx;
        return velocity_from_f(innerCell, innerX, y, parity);
      }
      return velocity_from_f(cell, x, y, parity);
    }

    return velocity_from_f(cell, x, y, parity);
  }

  @compute @workgroup_size(WGX, WGY, WGZ)
  fn render(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.Nx || gid.y >= P.Ny) { return; }
    let cell: u32 = gid.x + gid.y * P.Nx;

    let m = mask[cell];

    // Mask coloring
    if (is_solid(m)) {
      textureStore(outputTex, vec2<i32>(gid.xy), vec4<f32>(0.1, 0.1, 0.1, 1.0));
      return;
    }
    if (is_eq(m)) {
      textureStore(outputTex, vec2<i32>(gid.xy), vec4<f32>(1.0, 0.0, 0.0, 1.0));
      return;
    }

    let parity = Pd.parity;

    var s: f32;

    // mode: 0=|u|, 1=vorticity
    if (P.mode == 1u) {
      let maskX = P.Nx - 1u;
      let maskY = P.Ny - 1u;

      let xE = (gid.x + 1u) & maskX;
      let xW = (gid.x - 1u) & maskX;
      let yN = (gid.y + 1u) & maskY;
      let yS = (gid.y - 1u) & maskY;

      let cE = xE + gid.y * P.Nx;
      let cW = xW + gid.y * P.Nx;
      let cN = gid.x + yN * P.Nx;
      let cS = gid.x + yS * P.Nx;

      let vE = sample_velocity(cE, xE, gid.y, parity);
      let vW = sample_velocity(cW, xW, gid.y, parity);
      let vN = sample_velocity(cN, gid.x, yN, parity);
      let vS = sample_velocity(cS, gid.x, yS, parity);
      s = (vE.y - vW.y) * 0.5 - (vN.x - vS.x) * 0.5;
    } else {
      let u = velocity_from_f(cell, gid.x, gid.y, parity);
      let ux = u.x;
      let uy = u.y;
      s = sqrt(ux*ux + uy*uy);
    }

    let range = max(P.vmax - P.vmin, 1e-12);
    let t = clamp((s - P.vmin) / range, 0.0, 1.0);

    var rgb: vec3<f32>;
    if (P.cmap == 2u) {
      rgb = colormapRdBu(t);
    } else if (P.cmap == 1u) {
      rgb = colormapTurbo(t);
    } else {
      rgb = colormapViridis(t);
    }

    textureStore(outputTex, vec2<i32>(gid.xy), vec4<f32>(rgb, 1.0));
  }
