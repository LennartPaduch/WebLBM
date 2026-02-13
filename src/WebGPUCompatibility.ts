import {
  WEBGPU_CANIUSE_SNAPSHOT_DATE,
  WEBGPU_CANIUSE_URL,
  WEBGPU_COMPAT_TABLE,
  type BrowserKey,
  type CompatBand,
  type CompatEntry,
  type WebGPUCompatStatus,
} from "./WebGPUCompatData";

export type { WebGPUCompatStatus } from "./WebGPUCompatData";

type DetectedBrowser = {
  key: BrowserKey | "unknown";
  label: string;
  version: number | null;
};

type CompatAssessment = {
  detected: string;
  status: WebGPUCompatStatus;
  statusText: string;
  suggestion: string | null;
  source: string;
};

type CompatDataset = {
  table: Record<BrowserKey, CompatEntry>;
  snapshotDate: string;
  sourceUrl: string;
  mode: "snapshot" | "live";
};

type LiveBrowserMap = {
  sourceKey: string;
  key: BrowserKey;
  label: string;
};

const FALLBACK_DATASET: CompatDataset = {
  table: WEBGPU_COMPAT_TABLE,
  snapshotDate: WEBGPU_CANIUSE_SNAPSHOT_DATE,
  sourceUrl: WEBGPU_CANIUSE_URL,
  mode: "snapshot",
};

const LIVE_CANIUSE_SOURCES = [
  "https://cdn.jsdelivr.net/gh/Fyrd/caniuse@main/features-json/webgpu.json",
  "https://raw.githubusercontent.com/Fyrd/caniuse/main/features-json/webgpu.json",
] as const;
const LIVE_FETCH_TIMEOUT_MS = 1200;

const LIVE_BROWSER_MAP: LiveBrowserMap[] = [
  { sourceKey: "chrome", key: "chrome", label: "Chrome" },
  { sourceKey: "edge", key: "edge", label: "Edge" },
  { sourceKey: "firefox", key: "firefox", label: "Firefox" },
  { sourceKey: "safari", key: "safari", label: "Safari" },
  { sourceKey: "opera", key: "opera", label: "Opera" },
  { sourceKey: "ios_saf", key: "ios_safari", label: "iOS WebKit browsers" },
  {
    sourceKey: "and_chr",
    key: "chrome_android",
    label: "Chromium browsers on Android",
  },
  {
    sourceKey: "and_ff",
    key: "firefox_android",
    label: "Firefox for Android",
  },
  { sourceKey: "samsung", key: "samsung_internet", label: "Samsung Internet" },
  { sourceKey: "op_mob", key: "opera_mobile", label: "Opera Mobile" },
  { sourceKey: "android", key: "android_browser", label: "Android Browser" },
];

let liveDatasetPromise: Promise<CompatDataset | null> | null = null;

function extractVersion(ua: string, re: RegExp): number | null {
  const match = ua.match(re);
  if (!match?.[1]) return null;
  const v = parseFloat(match[1].replace(/_/g, "."));
  return Number.isFinite(v) ? v : null;
}

function detectBrowser(ua: string): DetectedBrowser {
  const isAndroid = /Android/i.test(ua);
  const isIOS = /\b(iPhone|iPad|iPod)\b/i.test(ua);

  if (isIOS) {
    const appLabel = /CriOS\//.test(ua)
      ? "Chrome on iOS"
      : /EdgiOS\//.test(ua)
        ? "Edge on iOS"
        : /FxiOS\//.test(ua)
          ? "Firefox on iOS"
          : "Safari on iOS";
    const version =
      extractVersion(ua, /Version\/([\d.]+)/i) ??
      extractVersion(ua, /OS (\d+(?:[_.]\d+)?)/i);
    return { key: "ios_safari", label: appLabel, version };
  }

  if (isAndroid) {
    const samsung = extractVersion(ua, /SamsungBrowser\/([\d.]+)/i);
    if (samsung !== null) {
      return { key: "samsung_internet", label: "Samsung Internet", version: samsung };
    }

    const operaMobile = extractVersion(ua, /OPR\/([\d.]+)/i);
    if (operaMobile !== null) {
      return { key: "opera_mobile", label: "Opera Mobile", version: operaMobile };
    }

    const firefoxAndroid = extractVersion(ua, /Firefox\/([\d.]+)/i);
    if (firefoxAndroid !== null) {
      return { key: "firefox_android", label: "Firefox for Android", version: firefoxAndroid };
    }

    const chromiumAndroid =
      extractVersion(ua, /EdgA\/([\d.]+)/i) ??
      extractVersion(ua, /Chrome\/([\d.]+)/i) ??
      extractVersion(ua, /Chromium\/([\d.]+)/i);
    if (chromiumAndroid !== null) {
      return {
        key: "chrome_android",
        label: "Chromium-based browser on Android",
        version: chromiumAndroid,
      };
    }

    const androidBrowser = extractVersion(ua, /Version\/([\d.]+)/i);
    if (androidBrowser !== null && /Mobile Safari/i.test(ua)) {
      return { key: "android_browser", label: "Android Browser", version: androidBrowser };
    }
  }

  const edge = extractVersion(ua, /Edg\/([\d.]+)/i);
  if (edge !== null) return { key: "edge", label: "Edge", version: edge };

  const opera = extractVersion(ua, /OPR\/([\d.]+)/i);
  if (opera !== null) return { key: "opera", label: "Opera", version: opera };

  const firefox = extractVersion(ua, /Firefox\/([\d.]+)/i);
  if (firefox !== null) return { key: "firefox", label: "Firefox", version: firefox };

  const chrome =
    extractVersion(ua, /Chrome\/([\d.]+)/i) ??
    extractVersion(ua, /Chromium\/([\d.]+)/i);
  if (chrome !== null) return { key: "chrome", label: "Chrome/Chromium", version: chrome };

  const safari = extractVersion(ua, /Version\/([\d.]+)/i);
  if (safari !== null && /Safari\//i.test(ua)) {
    return { key: "safari", label: "Safari", version: safari };
  }

  return { key: "unknown", label: "Unknown browser", version: null };
}

function statusLabel(status: WebGPUCompatStatus): string {
  if (status === "supported") return "supported";
  if (status === "partial") return "partially supported";
  if (status === "disabled-by-default") return "disabled by default";
  if (status === "not-supported") return "not supported";
  return "unknown";
}

function findBand(entry: CompatEntry, version: number): CompatBand | null {
  for (const band of entry.bands) {
    const inLower = version >= band.min;
    const inUpper = band.max === null || version <= band.max;
    if (inLower && inUpper) return band;
  }
  return null;
}

function minSupportedVersion(entry: CompatEntry): number | null {
  let min: number | null = null;
  for (const band of entry.bands) {
    if (band.status !== "supported" && band.status !== "partial") continue;
    if (min === null || band.min < min) min = band.min;
  }
  return min;
}

function buildSuggestion(
  detected: DetectedBrowser,
  status: WebGPUCompatStatus,
  entry: CompatEntry | null,
): string | null {
  if (!entry) {
    return "Try the latest stable Chrome, Edge, or Safari build with hardware acceleration enabled.";
  }

  const minSupported = minSupportedVersion(entry);

  if (status === "supported") {
    return null;
  }

  if (status === "partial") {
    return "This browser/version has partial WebGPU support. Update to the latest stable release for better compatibility.";
  }

  if (status === "disabled-by-default") {
    if (minSupported !== null) {
      return `Update ${entry.label} to version ${minSupported}+ if available, then retry.`;
    }
    return `${entry.label} lists WebGPU as disabled by default for this version range.`;
  }

  if (status === "not-supported") {
    if (minSupported !== null) {
      return `Update ${entry.label} to version ${minSupported}+ for WebGPU support.`;
    }
    return `${entry.label} is listed as not supporting WebGPU.`;
  }

  if (detected.key === "unknown") {
    return "Browser could not be identified from user agent. Try the latest stable Chrome, Edge, or Safari.";
  }

  return "Try updating the browser to the latest stable release.";
}

function parseSupportStatus(raw: unknown): WebGPUCompatStatus {
  const tokens = String(raw)
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((t) => t.toLowerCase());
  const codes = new Set(tokens.filter((t) => /^[a-z]$/.test(t)));

  if (codes.has("y")) return "supported";
  if (codes.has("a") || codes.has("x") || codes.has("p")) return "partial";
  if (codes.has("d")) return "disabled-by-default";
  if (codes.has("n")) return "not-supported";
  return "unknown";
}

function parseVersionNumber(value: string): number | null {
  const normalized = value.replace(/_/g, ".").trim();
  if (!/^\d+(\.\d+)?$/.test(normalized)) return null;
  const n = Number.parseFloat(normalized);
  return Number.isFinite(n) ? n : null;
}

function parseVersionRange(versionKey: string): { min: number; max: number | null } | null {
  const key = String(versionKey).trim();
  if (!key) return null;

  if (key.includes("-")) {
    const [startRaw, endRaw] = key.split("-", 2);
    const min = parseVersionNumber(startRaw);
    if (min === null) return null;
    const max = endRaw ? parseVersionNumber(endRaw) : null;
    return { min, max: max ?? min };
  }

  const single = parseVersionNumber(key);
  if (single === null) return null;
  return { min: single, max: single };
}

function mergeBands(bands: CompatBand[]): CompatBand[] {
  const sorted = [...bands].sort((a, b) => {
    if (a.min !== b.min) return a.min - b.min;
    const aMax = a.max ?? Number.POSITIVE_INFINITY;
    const bMax = b.max ?? Number.POSITIVE_INFINITY;
    return aMax - bMax;
  });

  const merged: CompatBand[] = [];
  for (const band of sorted) {
    const prev = merged[merged.length - 1];
    if (!prev) {
      merged.push({ ...band });
      continue;
    }

    const prevMax = prev.max ?? Number.POSITIVE_INFINITY;
    const overlaps = band.min <= prevMax;
    if (prev.status === band.status && overlaps) {
      if (prev.max === null || band.max === null) {
        prev.max = null;
      } else {
        prev.max = Math.max(prev.max, band.max);
      }
      continue;
    }

    merged.push({ ...band });
  }

  if (merged.length > 0) {
    merged[merged.length - 1].max = null;
  }

  return merged;
}

function parseBrowserStats(stats: unknown): CompatBand[] {
  if (!stats || typeof stats !== "object") return [];

  const bands: CompatBand[] = [];
  for (const [versionKey, rawStatus] of Object.entries(stats)) {
    const range = parseVersionRange(versionKey);
    if (!range) continue;
    bands.push({
      min: range.min,
      max: range.max,
      status: parseSupportStatus(rawStatus),
    });
  }

  return mergeBands(bands);
}

function parseSnapshotDate(payload: unknown): string {
  if (!payload || typeof payload !== "object") {
    return new Date().toISOString().slice(0, 10);
  }

  const candidate = (payload as { updated?: unknown }).updated;
  if (typeof candidate === "number" && Number.isFinite(candidate)) {
    const d = new Date(candidate * 1000);
    if (!Number.isNaN(d.getTime())) {
      return d.toISOString().slice(0, 10);
    }
  }

  return new Date().toISOString().slice(0, 10);
}

function parseLiveDataset(payload: unknown, sourceUrl: string): CompatDataset | null {
  const statsObj =
    payload && typeof payload === "object"
      ? ((payload as { stats?: unknown }).stats as Record<string, unknown> | undefined)
      : undefined;
  if (!statsObj || typeof statsObj !== "object") return null;

  const table = {} as Record<BrowserKey, CompatEntry>;
  for (const browser of LIVE_BROWSER_MAP) {
    const bands = parseBrowserStats(statsObj[browser.sourceKey]);
    table[browser.key] = {
      label: browser.label,
      bands: bands.length > 0 ? bands : WEBGPU_COMPAT_TABLE[browser.key].bands,
    };
  }

  return {
    table,
    sourceUrl,
    mode: "live",
    snapshotDate: parseSnapshotDate(payload),
  };
}

async function fetchWithTimeout(url: string, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, {
      signal: controller.signal,
      cache: "no-store",
    });
  } finally {
    window.clearTimeout(timeout);
  }
}

async function loadLiveDataset(): Promise<CompatDataset | null> {
  for (const url of LIVE_CANIUSE_SOURCES) {
    try {
      const res = await fetchWithTimeout(url, LIVE_FETCH_TIMEOUT_MS);
      if (!res.ok) continue;

      const payload = (await res.json()) as unknown;
      const dataset = parseLiveDataset(payload, url);
      if (dataset) return dataset;
    } catch {}
  }

  return null;
}

function getLiveDatasetPromise(): Promise<CompatDataset | null> {
  if (!liveDatasetPromise) {
    liveDatasetPromise = loadLiveDataset();
  }
  return liveDatasetPromise;
}

function formatSource(dataset: CompatDataset): string {
  if (dataset.mode === "live") {
    return `${WEBGPU_CANIUSE_URL} (live data via ${dataset.sourceUrl}, snapshot ${dataset.snapshotDate})`;
  }
  return `${WEBGPU_CANIUSE_URL} (snapshot ${dataset.snapshotDate})`;
}

function assessWithDataset(dataset: CompatDataset, ua: string): CompatAssessment {
  const detected = detectBrowser(ua);
  const entry = detected.key === "unknown" ? null : dataset.table[detected.key];

  let status: WebGPUCompatStatus = "unknown";
  if (entry && detected.version !== null) {
    const band = findBand(entry, detected.version);
    if (band) status = band.status;
  }

  const versionLabel = detected.version === null ? "?" : String(detected.version);
  const statusText = statusLabel(status);
  const suggestion = buildSuggestion(detected, status, entry);

  return {
    detected: `${detected.label} ${versionLabel}`,
    status,
    statusText,
    suggestion,
    source: formatSource(dataset),
  };
}

function detailsFromAssessment(info: CompatAssessment): string[] {
  const lines = [
    `Detected browser: ${info.detected}`,
    `Can I Use WebGPU status: ${info.statusText}`,
  ];
  if (info.suggestion) lines.push(info.suggestion);
  lines.push(`Compatibility source: ${info.source}`);
  return lines;
}

export function assessBrowserWebGPUCompatibility(
  ua: string = navigator.userAgent,
): CompatAssessment {
  return assessWithDataset(FALLBACK_DATASET, ua);
}

export async function assessBrowserWebGPUCompatibilityLive(
  ua: string = navigator.userAgent,
): Promise<CompatAssessment> {
  const liveDataset = await getLiveDatasetPromise();
  return assessWithDataset(liveDataset ?? FALLBACK_DATASET, ua);
}

export function getBrowserCompatibilityDetails(ua: string = navigator.userAgent): string[] {
  return detailsFromAssessment(assessBrowserWebGPUCompatibility(ua));
}

export async function getBrowserCompatibilityDetailsLive(
  ua: string = navigator.userAgent,
): Promise<string[]> {
  const assessment = await assessBrowserWebGPUCompatibilityLive(ua);
  return detailsFromAssessment(assessment);
}
