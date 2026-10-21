import type { AssessResult } from "./types";

const PLACEHOLDER =
  "data:image/svg+xml;utf8," +
  encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="667" viewBox="0 0 1000 667">' +
      '<rect width="1000" height="667" fill="#2a3036"/>' +
      '<rect x="120" y="200" width="760" height="300" rx="40" fill="#3b434a"/>' +
      '<rect x="300" y="240" width="300" height="140" fill="none" stroke="#F85149" stroke-width="6"/>' +
      "</svg>",
  );

export const fixtureResult: AssessResult = {
  triage: {
    decision: "total_loss_review",
    label: "Total-loss review",
    n_instances: 18,
    n_severe: 10,
    n_moderate: 7,
    n_minor: 1,
    total_damaged_area_px: 242470,
  },
  instances: [
  { damage: "glass shatter", panel: "windshield", severity: "severe", confidence: 0.87, mask_area_px: 30000, source_model: "cardd" },
  { damage: "dent", panel: null, severity: "severe", confidence: 0.42, mask_area_px: 22000, source_model: "cardd" },
  { damage: "dent", panel: null, severity: "severe", confidence: 0.37, mask_area_px: 18000, source_model: "cardd" },
  { damage: "dent", panel: null, severity: "severe", confidence: 0.30, mask_area_px: 16000, source_model: "cardd" },
  { damage: "torn", panel: null, severity: "severe", confidence: 0.22, mask_area_px: 15000, source_model: "cardd" },
  { damage: "dent", panel: null, severity: "severe", confidence: 0.21, mask_area_px: 14000, source_model: "cardd" },
  { damage: "missing_part", panel: "door", severity: "severe", confidence: 0.17, mask_area_px: 20000, source_model: "cardd" },
  { damage: "missing_part", panel: "rear_bumper", severity: "severe", confidence: 0.09, mask_area_px: 9000, source_model: "cardd" },
  { damage: "torn", panel: null, severity: "severe", confidence: 0.09, mask_area_px: 8000, source_model: "cardd" },
  { damage: "missing_part", panel: null, severity: "severe", confidence: 0.07, mask_area_px: 7000, source_model: "cardd" },
  { damage: "dent", panel: null, severity: "moderate", confidence: 0.67, mask_area_px: 16000, source_model: "cardd" },
  { damage: "dent", panel: "door", severity: "moderate", confidence: 0.46, mask_area_px: 15000, source_model: "cardd" },
  { damage: "dent", panel: null, severity: "moderate", confidence: 0.31, mask_area_px: 10000, source_model: "cardd" },
  { damage: "torn", panel: null, severity: "moderate", confidence: 0.30, mask_area_px: 9000, source_model: "cardd" },
  { damage: "torn", panel: null, severity: "moderate", confidence: 0.17, mask_area_px: 6000, source_model: "cardd" },
  { damage: "scratch", panel: null, severity: "moderate", confidence: 0.16, mask_area_px: 5000, source_model: "cardd" },
  { damage: "torn", panel: null, severity: "moderate", confidence: 0.09, mask_area_px: 4000, source_model: "cardd" },
  { damage: "torn", panel: null, severity: "minor", confidence: 0.38, mask_area_px: 18470, source_model: "cardd" },
  ],
  panels: { bonnet: 0, front_bumper: 0, rear_bumper: 1, door: 2, headlamp: 0, windshield: 1 },
  unassigned: 14,
  overlay: PLACEHOLDER,
  image: { width: 1000, height: 667 },
  sensitivity: "high_recall",
  secondary_conf: 0.05,
  elapsed_ms: 8400,
};
