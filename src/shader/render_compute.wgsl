  struct VisParams {
    Nx:        u32,
    Ny:        u32,
    cellCount: u32,
    mode:      u32, // 0=|u|, 1=vorticity
    cmap:      u32, // 0=Viridis, 1=Turbo, 2=RdBu
    vmin:      f32,
    vmax:      f32,
  };

  @group(0) @binding(0) var<storage, read>  global_u   : array<f32>;   // [0..C)=ux, [C..2C)=uy
  @group(0) @binding(1) var<storage, read>  mask       : array<u32>;
  @group(0) @binding(2) var<uniform>        P          : VisParams;
  @group(0) @binding(3) var outputTex : texture_storage_2d<rgba8unorm, write>;

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

  fn sample_lut10(t: f32, lut: ptr<function, array<vec3<f32>,10>>) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0) * 9.0;
    let i = u32(floor(x));
    let j = min(i + 1u, 9u);
    let u = smoothstep(0.0, 1.0, fract(x));
    return mix((*lut)[i], (*lut)[j], u);
  }

  fn colormapViridis(t: f32) -> vec3<f32> {
    var lut = VIRIDIS_LUT;
    return sample_lut10(t, &lut);
  }
  fn colormapTurbo(t: f32) -> vec3<f32> {
    var lut = TURBO_LUT;
    return sample_lut10(t, &lut);
  }

  // ---- Macro access from global fields ----
  fn read_u(cell: u32) -> vec2<f32> {
    if (is_solid(mask[cell])) { return vec2<f32>(0.0); }
    let C = P.cellCount;
    let ux = global_u[cell];
    let uy = global_u[cell+C];
    return vec2<f32>(ux, uy);
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

    let u   = read_u(cell);
    let ux  = u.x;
    let uy  = u.y;

    var s: f32;

    // mode: 0=|u|, 1=vorticity
    if (P.mode == 1u) {
      let j = get_neighbors(gid.x, gid.y);
      let vE = read_u(j[1]);
      let vW = read_u(j[2]);
      let vN = read_u(j[3]);
      let vS = read_u(j[4]);
      s = (vE.y - vW.y) * 0.5 - (vN.x - vS.x) * 0.5;
    } else {
      s = sqrt(ux*ux + uy*uy);
    }

    let range = max(P.vmax - P.vmin, 1e-12);
    let t = clamp((s - P.vmin) / range, 0.0, 1.0);

    var rgb: vec3<f32>;
    if (P.cmap == 2u) {
      rgb = colormapRdBu(t);
    } else {
      rgb = select(colormapViridis(t), colormapTurbo(t), P.cmap == 1u);
    }

    textureStore(outputTex, vec2<i32>(gid.xy), vec4<f32>(rgb, 1.0));
  }
