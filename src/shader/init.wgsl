    struct Params {
      Nx:u32,
      Ny:u32, 
      Q:u32,
      inletUx: f32,
      inletUy: f32
    };

    @group(0) @binding(0) var<storage, read_write> f           : array<f16>;   
    @group(0) @binding(1) var<storage, read>       mask        : array<u32>;
    @group(0) @binding(2) var<uniform>             P           : Params;


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
          // Outlet starts at rest equilibrium.
          let feq = feq_d2q9_shifted(1.0, vec2<f32>(0.0, 0.0));
          for (var d:u32 = 0u; d < 9u; d++) { 
            f[addr(d, cell, C)] = pack_f16s(feq[d]); 
          }
          return;
        }
        let feq = feq_d2q9_shifted(1.0, vec2<f32>(P.inletUx, P.inletUy));
        for (var d:u32 = 0u; d < 9u; d++) { 
          f[addr(d, cell, C)] = pack_f16s(feq[d]); 
        }
        return;
      }

      // Build equilibrium with shifted DDFs (ρ starts at ~1.0) (Skordos, 1993, https://arxiv.org/abs/comp-gas/9306002)
      let feq = feq_d2q9_shifted(r, vec2<f32>(ux, uy));

      for (var d:u32 = 0u; d < 9u; d++) {
        f[addr(d, cell, C)] = pack_f16s(feq[d]);
      }
    }
