// Snapshot generated from Can I Use WebGPU data.

export type BrowserKey =
  | "chrome"
  | "edge"
  | "firefox"
  | "safari"
  | "opera"
  | "ios_safari"
  | "chrome_android"
  | "firefox_android"
  | "samsung_internet"
  | "opera_mobile"
  | "android_browser";

export type WebGPUCompatStatus =
  | "supported"
  | "partial"
  | "disabled-by-default"
  | "not-supported"
  | "unknown";

export type CompatBand = {
  min: number;
  max: number | null;
  status: WebGPUCompatStatus;
};

export type CompatEntry = {
  label: string;
  bands: CompatBand[];
};

export const WEBGPU_CANIUSE_SNAPSHOT_DATE = "2026-02-12";
export const WEBGPU_CANIUSE_URL = "https://caniuse.com/webgpu";

export const WEBGPU_COMPAT_TABLE: Record<BrowserKey, CompatEntry> = {
  chrome: {
    label: "Chrome",
    bands: [
      { min: 0, max: 79, status: "not-supported" },
      { min: 80, max: 112, status: "disabled-by-default" },
      { min: 113, max: null, status: "supported" },
    ],
  },
  edge: {
    label: "Edge",
    bands: [
      { min: 0, max: 79, status: "not-supported" },
      { min: 80, max: 112, status: "disabled-by-default" },
      { min: 113, max: null, status: "supported" },
    ],
  },
  firefox: {
    label: "Firefox",
    bands: [
      { min: 0, max: 62, status: "not-supported" },
      { min: 63, max: null, status: "disabled-by-default" },
    ],
  },
  safari: {
    label: "Safari",
    bands: [
      { min: 0, max: 17.3, status: "not-supported" },
      { min: 17.4, max: 18.6, status: "disabled-by-default" },
      { min: 26, max: null, status: "partial" },
    ],
  },
  opera: {
    label: "Opera",
    bands: [
      { min: 0, max: 72, status: "not-supported" },
      { min: 73, max: 98, status: "disabled-by-default" },
      { min: 99, max: null, status: "supported" },
    ],
  },
  ios_safari: {
    label: "iOS WebKit browsers",
    bands: [
      { min: 0, max: 17.3, status: "not-supported" },
      { min: 17.4, max: 18.7, status: "disabled-by-default" },
      { min: 26, max: null, status: "supported" },
    ],
  },
  chrome_android: {
    label: "Chromium browsers on Android",
    bands: [
      { min: 0, max: 143, status: "not-supported" },
      { min: 144, max: null, status: "supported" },
    ],
  },
  firefox_android: {
    label: "Firefox for Android",
    bands: [
      { min: 0, max: 146, status: "not-supported" },
      { min: 147, max: null, status: "disabled-by-default" },
    ],
  },
  samsung_internet: {
    label: "Samsung Internet",
    bands: [
      { min: 0, max: 23, status: "not-supported" },
      { min: 24, max: null, status: "supported" },
    ],
  },
  opera_mobile: {
    label: "Opera Mobile",
    bands: [
      { min: 0, max: 79, status: "not-supported" },
      { min: 80, max: null, status: "supported" },
    ],
  },
  android_browser: {
    label: "Android Browser",
    bands: [{ min: 0, max: null, status: "not-supported" }],
  },
};
