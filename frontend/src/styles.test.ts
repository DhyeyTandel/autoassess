import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";

const css = readFileSync("src/styles.css", "utf8");

describe("styles.css", () => {
  it("does not theme-qualify .region selectors (would out-specify .region.on)", () => {
    const selectors = css.match(/[^{}]+(?=\{)/g) ?? [];
    const bad = selectors.filter((s) => s.includes(".region") && s.includes("[data-theme"));
    expect(bad).toEqual([]);
  });
  it("defines region tokens per theme", () => {
    expect(css.match(/--region-fill:/g)?.length).toBe(2);
    expect(css.match(/--region-stroke:/g)?.length).toBe(2);
  });
});
