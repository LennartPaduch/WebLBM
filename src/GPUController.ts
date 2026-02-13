import { AppError } from "./AppError";
import {
  getBrowserCompatibilityDetails,
  getBrowserCompatibilityDetailsLive,
} from "./WebGPUCompatibility";

type AdapterInfoLike = {
  vendor?: string;
  architecture?: string;
  device?: string;
  description?: string;
  backend?: string;
  driver?: string;
  type?: string;
  isFallbackAdapter?: boolean;
};

function toErrorText(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

function readAdapterInfo(adapter: GPUAdapter): string | null {
  const info = (adapter as GPUAdapter & { info?: AdapterInfoLike }).info;
  if (!info) return null;

  const parts = [info.vendor, info.architecture, info.device, info.description]
    .map((v) => (typeof v === "string" ? v.trim() : ""))
    .filter(Boolean);

  return parts.length ? parts.join(" | ") : null;
}

function readAdapterDiagnostics(adapter: GPUAdapter): {
  isFallback: boolean;
  type: string | null;
  backend: string | null;
} {
  const info = (adapter as GPUAdapter & { info?: AdapterInfoLike }).info;
  if (!info) {
    return { isFallback: false, type: null, backend: null };
  }

  const type =
    typeof info.type === "string" && info.type.trim() ? info.type.trim() : null;
  const backend =
    typeof info.backend === "string" && info.backend.trim()
      ? info.backend.trim()
      : null;
  const isFallback = info.isFallbackAdapter === true || type?.toLowerCase() === "cpu";
  return { isFallback, type, backend };
}

function summarizeAdapterFeatures(features: GPUSupportedFeatures): string {
  const list = [...features].sort();
  if (list.length === 0) return "Adapter features: (none)";

  const shownCount = 8;
  const shown = list.slice(0, shownCount);
  const remaining = list.length - shown.length;
  const suffix = remaining > 0 ? `, +${remaining} more` : "";
  return `Adapter features (${list.length}): ${shown.join(", ")}${suffix}`;
}

function isLikelySoftwareAdapter(adapterInfo: string): boolean {
  const text = adapterInfo.toLowerCase();
  return (
    text.includes("swiftshader") || text.includes("llvmpipe") || text.includes("software")
  );
}

export class GPUController {
  adapter: GPUAdapter;
  device: GPUDevice;

  context!: GPUCanvasContext;
  contextFormat: GPUTextureFormat;

  private constructor(
    adapter: GPUAdapter,
    device: GPUDevice,
    format: GPUTextureFormat,
  ) {
    this.adapter = adapter;
    this.device = device;
    this.contextFormat = format;
  }

  static async create(): Promise<GPUController> {
    let compatibilityDetailsPromise: Promise<string[]> | null = null;
    const getCompatibilityDetails = () => getBrowserCompatibilityDetails();
    const getLiveCompatibilityDetails = () => {
      if (!compatibilityDetailsPromise) {
        compatibilityDetailsPromise = getBrowserCompatibilityDetailsLive().catch(() => []);
      }
      return compatibilityDetailsPromise;
    };
    const createCompatibilityError = (message: string, details: string[]) =>
      new AppError(message, {
        kind: "webgpu-compat",
        details,
        liveDetails: getLiveCompatibilityDetails(),
      });

    if (!navigator.gpu) {
      const details = [
        "This browser does not expose `navigator.gpu`.",
        "Use a WebGPU-enabled browser build and make sure hardware acceleration is enabled.",
        ...getCompatibilityDetails(),
      ];
      if (!window.isSecureContext) {
        details.unshift(
          "The page is not running in a secure context. WebGPU requires HTTPS (or localhost).",
          "Load this site over HTTPS, or use http://localhost during local development.",
        );
        throw createCompatibilityError(
          "WebGPU requires a secure context (HTTPS or localhost).",
          details,
        );
      }
      throw createCompatibilityError("WebGPU is not available in this browser.", details);
    }

    let adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });

    // Retry without a power hint if high-performance selection fails.
    if (!adapter) adapter = await navigator.gpu.requestAdapter();

    if (!adapter) {
      throw createCompatibilityError("No suitable WebGPU adapter was found.", [
        "The browser exposed WebGPU, but could not create a GPU adapter.",
        "Common causes: hardware acceleration disabled, blocked/outdated GPU driver, or VM/remote desktop sessions.",
        "On Linux, ensure Vulkan userspace drivers are installed and accessible to the browser.",
        ...getCompatibilityDetails(),
      ]);
    }

    if (!adapter.features.has("shader-f16")) {
      const details = [
        "This app requires `shader-f16` for FP16 distribution storage.",
        "WebGPU being hardware accelerated does not guarantee `shader-f16` support.",
        "This app currently has no non-FP16 fallback mode.",
        "Try the latest stable Chrome or Edge on desktop, then restart the browser.",
        "On Linux, ensure Vulkan userspace drivers are installed and avoid software-rendering sessions.",
        "Open chrome://gpu and confirm WebGPU is hardware accelerated (not SwiftShader/software).",
        summarizeAdapterFeatures(adapter.features),
        ...getCompatibilityDetails(),
      ];
      const adapterInfo = readAdapterInfo(adapter);
      if (adapterInfo) {
        details.unshift(`Detected adapter: ${adapterInfo}`);
        if (isLikelySoftwareAdapter(adapterInfo)) {
          details.splice(
            1,
            0,
            "Detected software adapter (SwiftShader/llvmpipe). Chrome is not using your hardware GPU for WebGPU.",
            "Disable software-rendering flags, keep hardware acceleration enabled, and restart Chrome.",
            "If launching from terminal, remove flags like `--disable-gpu` or `--use-angle=swiftshader`.",
            "If needed on Linux, try `--enable-features=Vulkan --use-angle=vulkan` and recheck chrome://gpu.",
          );
        }
      }
      const adapterDiag = readAdapterDiagnostics(adapter);
      if (adapterDiag.isFallback) {
        details.splice(
          1,
          0,
          "Adapter reports fallback/CPU mode for this page. This app needs a hardware GPU adapter with `shader-f16`.",
        );
      }
      if (adapterDiag.backend) {
        details.push(`WebGPU backend: ${adapterDiag.backend}`);
      }
      if (adapterDiag.type) {
        details.push(`Adapter type: ${adapterDiag.type}`);
      }

      throw createCompatibilityError(
        "GPU adapter found, but `shader-f16` is not supported.",
        details,
      );
    }

    const limits = adapter.limits;
    let device: GPUDevice;
    try {
      device = await adapter.requestDevice({
        requiredFeatures: ["shader-f16"],
        requiredLimits: {
          maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
          maxBufferSize: limits.maxBufferSize,
          maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
          maxComputeWorkgroupSizeY: limits.maxComputeWorkgroupSizeY,
          maxComputeWorkgroupsPerDimension:
            limits.maxComputeWorkgroupsPerDimension,
          maxComputeInvocationsPerWorkgroup:
            limits.maxComputeInvocationsPerWorkgroup,
          maxTextureDimension2D: limits.maxTextureDimension2D,
        },
      });
    } catch (err) {
      const details = [
        `requestDevice failed: ${toErrorText(err)}`,
        "The browser rejected the requested feature/limit set for this adapter.",
        ...getCompatibilityDetails(),
      ];
      const adapterInfo = readAdapterInfo(adapter);
      if (adapterInfo) details.unshift(`Detected adapter: ${adapterInfo}`);

      throw createCompatibilityError("Failed to create a WebGPU device.", details);
    }

    device.onuncapturederror = (e) => {
      console.error(`WebGPU uncaptured error: ${e.error?.message ?? e}`);
    };

    device.lost.then((info) => {
      console.error("WebGPU device lost:", info);
    });

    const format = navigator.gpu.getPreferredCanvasFormat();
    return new GPUController(adapter, device, format);
  }

  configureCanvas(canvas: HTMLCanvasElement) {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));

    canvas.width = w;
    canvas.height = h;

    const ctx = canvas.getContext("webgpu") as GPUCanvasContext | null;
    if (!ctx) {
      throw new AppError('Failed to get "webgpu" canvas context.', {
        kind: "webgpu-compat",
        details: [
          "The browser may expose WebGPU but still fail context creation for this canvas.",
          "Verify hardware acceleration is enabled and no extension/policy is disabling WebGPU rendering.",
          ...getBrowserCompatibilityDetails(),
        ],
        liveDetails: getBrowserCompatibilityDetailsLive().catch(() => []),
      });
    }

    this.context = ctx;

    ctx.configure({
      device: this.device,
      format: this.contextFormat,
      alphaMode: "premultiplied",
    });
  }
}
