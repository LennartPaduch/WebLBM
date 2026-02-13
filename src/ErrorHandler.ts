import { AppError, type AppErrorKind } from "./AppError";

type ParsedError = {
  message: string;
  details: string[];
  kind: AppErrorKind;
  liveDetails: Promise<string[]> | null;
};

const TECHNICAL_PREFIXES = [
  "Detected adapter:",
  "Adapter features",
  "Detected browser:",
  "Can I Use WebGPU status:",
  "Compatibility source:",
  "requestDevice failed:",
];

const LIVE_REFRESH_PREFIXES = [
  "Detected browser:",
  "Can I Use WebGPU status:",
  "Compatibility source:",
];

function parseError(input: unknown): ParsedError {
  if (input instanceof AppError) {
    return {
      message: input.message,
      details: input.details,
      kind: input.kind,
      liveDetails: input.liveDetails,
    };
  }

  if (typeof input === "string") {
    return {
      message: input,
      details: [],
      kind: "general",
      liveDetails: null,
    };
  }

  if (input instanceof Error) {
    return {
      message: input.message,
      details: [],
      kind: "general",
      liveDetails: null,
    };
  }

  return {
    message: "Something went wrong.",
    details: [],
    kind: "general",
    liveDetails: null,
  };
}

function isTechnicalDetail(detail: string): boolean {
  return TECHNICAL_PREFIXES.some((prefix) => detail.startsWith(prefix));
}

function detailKey(detail: string): string {
  for (const prefix of LIVE_REFRESH_PREFIXES) {
    if (detail.startsWith(prefix)) return prefix;
  }
  return detail;
}

function upsertDetail(
  detailList: HTMLUListElement,
  entriesByKey: Map<string, HTMLLIElement>,
  detail: string,
) {
  const normalized = detail.trim();
  if (!normalized) return;

  const key = detailKey(normalized);
  const existing = entriesByKey.get(key);
  if (existing) {
    existing.textContent = normalized;
    return;
  }

  const li = document.createElement("li");
  li.textContent = normalized;
  li.className = "break-words";
  detailList.appendChild(li);
  entriesByKey.set(key, li);
  detailList.classList.remove("hidden");
}

export function showError(message: unknown) {
  document.getElementById("wrapper")?.classList.add("hidden");

  const root = document.createElement("div");
  root.className = "fixed inset-0 z-50 overflow-auto bg-black/40 p-4 sm:p-6";

  const wrapper = document.createElement("div");
  wrapper.className = "mx-auto flex w-fit max-w-none flex-col gap-3";

  const card = document.createElement("div");
  card.setAttribute("role", "alert");
  card.className =
    "rounded-2xl border border-zinc-200 bg-white shadow-xl " +
    "dark:border-zinc-800 dark:bg-zinc-900";

  const parsed = parseError(message);

  card.innerHTML = `
    <div class="p-4 sm:p-5">
      <div class="flex items-start gap-3">
        <span class="mt-0.5 inline-flex h-6 w-6 items-center justify-center rounded-full
          bg-red-100 text-red-600 dark:bg-red-900/30">!</span>
        <div class="min-w-0 flex-1">
          <p class="text-sm font-medium text-zinc-800 dark:text-zinc-100 break-words"></p>
          <p class="mt-3 hidden text-[11px] font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400"
            data-action-title>What to try</p>
          <ul class="mt-1 list-disc space-y-1 pl-5 text-xs text-zinc-700 dark:text-zinc-200 hidden"
            data-action-list></ul>
          <details class="mt-3 hidden" data-tech-section>
            <summary class="cursor-pointer select-none text-xs font-medium text-zinc-600 dark:text-zinc-300">
              Technical details
            </summary>
            <ul class="mt-2 list-disc space-y-1 pl-5 text-xs text-zinc-500 dark:text-zinc-400 hidden"
              data-tech-list></ul>
          </details>
        </div>
      </div>
    </div>
  `;

  const p = card.querySelector("p") as HTMLParagraphElement;
  const actionTitle = card.querySelector("[data-action-title]") as HTMLParagraphElement;
  const actionList = card.querySelector("[data-action-list]") as HTMLUListElement;
  const techSection = card.querySelector("[data-tech-section]") as HTMLDetailsElement;
  const techList = card.querySelector("[data-tech-list]") as HTMLUListElement;
  p.textContent = parsed.message;

  const actionEntries = new Map<string, HTMLLIElement>();
  const techEntries = new Map<string, HTMLLIElement>();

  const refreshLayout = () => {
    if (actionList.childElementCount > 0) {
      actionTitle.classList.remove("hidden");
    } else {
      actionTitle.classList.add("hidden");
    }

    if (techList.childElementCount > 0) {
      techSection.classList.remove("hidden");
    } else {
      techSection.classList.add("hidden");
      techSection.open = false;
    }
  };

  const append = (detail: string) => {
    if (isTechnicalDetail(detail)) {
      upsertDetail(techList, techEntries, detail);
    } else {
      upsertDetail(actionList, actionEntries, detail);
    }
    refreshLayout();
  };

  for (const detail of parsed.details) {
    append(detail);
  }

  if (parsed.liveDetails) {
    void parsed.liveDetails
      .then((details) => {
        for (const detail of details) {
          append(detail);
        }
      })
      .catch(() => {});
  }

  wrapper.appendChild(card);

  if (parsed.kind === "webgpu-compat") {
    const support = document.createElement("section");
    support.className =
      "w-fit rounded-xl border border-zinc-200 bg-white p-3 shadow-md dark:border-zinc-800 dark:bg-zinc-900";

    const title = document.createElement("p");
    title.className = "text-xs font-medium text-zinc-700 dark:text-zinc-200";
    title.textContent = "WebGPU browser support chart";

    const link = document.createElement("a");
    link.href = "https://caniuse.com/webgpu";
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.className = "mt-2 inline-block text-xs text-blue-600 hover:underline dark:text-blue-400";
    link.textContent = "Open full WebGPU support table";

    const imageLink = document.createElement("a");
    imageLink.href = "https://caniuse.bitsofco.de/image/webgpu.png";
    imageLink.target = "_blank";
    imageLink.rel = "noopener noreferrer";
    imageLink.title = "Open chart image in a new tab";
    imageLink.className = "mt-2 inline-block cursor-zoom-in";

    const supportedBrowsersImg = document.createElement("img");
    supportedBrowsersImg.src = "https://caniuse.bitsofco.de/image/webgpu.png";
    supportedBrowsersImg.alt = "Can I Use support table for WebGPU";
    supportedBrowsersImg.className =
      "block h-auto w-auto max-w-none rounded-md border border-zinc-200 bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-950";
    supportedBrowsersImg.loading = "lazy";

    const hint = document.createElement("p");
    hint.className = "mt-1 text-[11px] text-zinc-500 dark:text-zinc-400";
    hint.textContent = "Click chart image to open full-size in a new tab.";

    imageLink.appendChild(supportedBrowsersImg);
    support.appendChild(title);
    support.appendChild(link);
    support.appendChild(imageLink);
    support.appendChild(hint);
    wrapper.appendChild(support);
  }

  root.appendChild(wrapper);
  document.body.appendChild(root);
}
