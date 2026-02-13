/// <reference types="@webgpu/types" />

// allow imports like `import s from './foo.wgsl?raw'`
declare module "*?raw" {
  const src: string;
  export default src;
}

