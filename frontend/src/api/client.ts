import type { AssessResult, Health, Sensitivity } from "./types";

export class ApiError extends Error {
  constructor(message: string, readonly status: number | null) {
    super(message);
    this.name = "ApiError";
  }
}

export const NETWORK_MESSAGE = "Couldn't reach the server. Try again.";

function messageForStatus(status: number): string {
  if (status === 413) return "That file is larger than 15 MB.";
  if (status === 400 || status === 422)
    return "That file couldn't be read as a photo. Try a JPEG, PNG, WebP or HEIC image.";
  if (status >= 500) return "The server hit a problem assessing this photo. Try again.";
  return "Something went wrong. Try again.";
}

export async function assess(file: File, sensitivity: Sensitivity): Promise<AssessResult> {
  const body = new FormData();
  body.append("image", file);
  body.append("sensitivity", sensitivity);
  let res: Response;
  try {
    res = await fetch("/api/assess", { method: "POST", body });
  } catch {
    throw new ApiError(NETWORK_MESSAGE, null);
  }
  if (!res.ok) throw new ApiError(messageForStatus(res.status), res.status);
  try {
    return (await res.json()) as AssessResult;
  } catch {
    throw new ApiError(messageForStatus(500), res.status);
  }
}

export async function health(): Promise<Health> {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) return { status: "unknown", busy: false };
    return (await res.json()) as Health;
  } catch {
    return { status: "unreachable", busy: false };
  }
}
