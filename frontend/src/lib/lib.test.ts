import { describe, expect, it } from "vitest";
import { panelRegions } from "./panels";
import { initialTheme } from "./theme";
import { formatConfidence, formatInt, prettifyLabel } from "./format";

describe("panelRegions", () => {
  it("activates only panels with count > 0", () => {
    const r = panelRegions({ door: 2, windshield: 1, bonnet: 0 });
    const active = r.filter((x) => x.active).map((x) => x.key).sort();
    expect(active).toEqual(["door_fl", "door_fr", "door_rl", "door_rr", "windshield"]);
  });
  it("maps headlamp to two corner rects and keeps neutral parts inactive", () => {
    const r = panelRegions({ headlamp: 1, bonnet: 3 });
    expect(r.filter((x) => x.panel === "headlamp")).toHaveLength(2);
    expect(r.filter((x) => x.panel === null).every((x) => !x.active)).toBe(true);
  });
  it("all-zero gives nothing active", () => {
    expect(panelRegions({}).some((x) => x.active)).toBe(false);
  });
});

describe("initialTheme", () => {
  it("prefers stored value", () => {
    expect(initialTheme("light", true)).toBe("light");
    expect(initialTheme("dark", false)).toBe("dark");
  });
  it("falls back to system preference", () => {
    expect(initialTheme(null, true)).toBe("dark");
    expect(initialTheme(null, false)).toBe("light");
    expect(initialTheme("bogus", false)).toBe("light");
  });
});

describe("format", () => {
  it("prettifies labels", () => {
    expect(prettifyLabel("missing_part")).toBe("Missing part");
    expect(prettifyLabel("glass shatter")).toBe("Glass shatter");
    expect(prettifyLabel("dent")).toBe("Dent");
  });
  it("formats numbers", () => {
    expect(formatConfidence(0.4)).toBe("0.40");
    expect(formatConfidence(0.866)).toBe("0.87");
    expect(formatInt(242470)).toBe("242,470");
  });
});
