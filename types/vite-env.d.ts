/// <reference types="@webgpu/types" />

// allow imports like `import s from './foo.wgsl?raw'`
declare module "*?raw" {
  const src: string;
  export default src;
}

// optional: allow plain `.wgsl` imports if you prefer plugin-based imports
declare module "*.wgsl" {
  const src: string;
  export default src;
}
