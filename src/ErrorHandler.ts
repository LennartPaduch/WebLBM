export function showError(message: unknown) {
  document.getElementById("wrapper")?.classList.add("hidden");

  const root = document.createElement("div");
  root.className = "fixed inset-0 z-50 grid place-items-center p-4";

  const wrapper = document.createElement("div");
  wrapper.className = "flex flex-col";

  const card = document.createElement("div");
  card.setAttribute("role", "alert");
  card.className =
    "w-full rounded-2xl border border-zinc-200 bg-white shadow-xl " +
    "dark:border-zinc-800 dark:bg-zinc-900";

  let safe =
    typeof message === "string"
      ? message
      : message instanceof Error
        ? message.message
        : "Something went wrong.";

  card.innerHTML = `
    <div class="p-4 sm:p-5">
      <div class="flex items-start gap-3">
        <span class="mt-0.5 inline-flex h-6 w-6 items-center justify-center rounded-full
          bg-red-100 text-red-600 dark:bg-red-900/30">!</span>
        <p class="text-sm text-zinc-700 dark:text-zinc-200 break-words"></p>
      </div>
    </div>
  `;

  const p = card.querySelector("p") as HTMLParagraphElement;
  p.textContent = String(safe);
  p.appendChild(document.createElement("br"));
  p.append("WebGPU Browser Support:");

  const supportedBrowsersImg = document.createElement("img");
  supportedBrowsersImg.src = "https://caniuse.bitsofco.de/image/webgpu.png";

  wrapper.appendChild(card);
  wrapper.appendChild(supportedBrowsersImg);

  root.appendChild(wrapper);
  document.body.appendChild(root);
}
