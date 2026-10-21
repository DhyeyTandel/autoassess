export type Decision = "total_loss_review" | "human_review" | "auto_approve";
export type Severity = "severe" | "moderate" | "minor";
export type Sensitivity = "high_recall" | "standard";
export type PanelName =
  | "bonnet"
  | "front_bumper"
  | "rear_bumper"
  | "door"
  | "headlamp"
  | "windshield";

export interface Triage {
  decision: Decision;
  label: string;
  n_instances: number;
  n_severe: number;
  n_moderate: number;
  n_minor: number;
  total_damaged_area_px: number;
}

export interface Instance {
  damage: string;
  panel: string | null;
  severity: Severity;
  confidence: number;
  mask_area_px: number;
  source_model: string;
}

export type PanelCounts = Record<PanelName, number>;

export interface AssessResult {
  triage: Triage;
  instances: Instance[];
  panels: PanelCounts;
  unassigned: number;
  overlay: string;
  image: { width: number; height: number };
  sensitivity: string;
  secondary_conf: number;
  elapsed_ms: number;
}

export interface Health {
  status: string;
  busy: boolean;
}
