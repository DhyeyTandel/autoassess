import { afterEach, describe, expect, it, vi } from "vitest";
import { assess, ApiError } from "./client";
import { fixtureResult } from "./fixture";

const file = new File(["x"], "a.jpg", { type: "image/jpeg" });
afterEach(() => vi.unstubAllGlobals());

describe("assess", () => {
  it("returns parsed JSON on 200 and posts multipart", async () => {
    const f = vi.fn().mockResolvedValue(new Response(JSON.stringify(fixtureResult), { status: 200 }));
    vi.stubGlobal("fetch", f);
    const r = await assess(file, "standard");
    expect(r.triage.n_instances).toBe(18);
    const [url, init] = f.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/assess");
    const body = init.body as FormData;
    expect(body.get("sensitivity")).toBe("standard");
    expect(body.get("image")).toBeInstanceOf(File);
  });
  it("maps 413 to a size message", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("{}", { status: 413 })));
    await expect(assess(file, "high_recall")).rejects.toMatchObject({
      message: "That file is larger than 15 MB.",
      status: 413,
    });
  });
  it("maps network failure", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("fail")));
    const err = await assess(file, "high_recall").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).message).toBe("Couldn't reach the server. Try again.");
  });
});
