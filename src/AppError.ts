export type AppErrorKind = "general" | "webgpu-compat";

type AppErrorOptions = {
  details?: string[];
  kind?: AppErrorKind;
  liveDetails?: Promise<string[]>;
};

export class AppError extends Error {
  readonly details: string[];
  readonly kind: AppErrorKind;
  readonly liveDetails: Promise<string[]> | null;

  constructor(message: string, detailsOrOptions: string[] | AppErrorOptions = []) {
    super(message);
    this.name = "AppError";

    if (Array.isArray(detailsOrOptions)) {
      this.details = detailsOrOptions;
      this.kind = "general";
      this.liveDetails = null;
      return;
    }

    this.details = detailsOrOptions.details ?? [];
    this.kind = detailsOrOptions.kind ?? "general";
    this.liveDetails = detailsOrOptions.liveDetails ?? null;
  }
}
