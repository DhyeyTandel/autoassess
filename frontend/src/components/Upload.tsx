import { useEffect, useRef, useState } from "react";
import type { Sensitivity } from "../api/types";
import { UploadIcon } from "./Icons";

const ACCEPT = "image/jpeg,image/png,image/webp,image/heic,image/heif,.heic,.heif";

interface Props {
  file: File | null;
  onFile: (f: File | null) => void;
  sensitivity: Sensitivity;
  onSensitivity: (s: Sensitivity) => void;
  onSubmit: () => void;
  running: boolean;
  waiting: boolean;
  seconds: number;
  error: string | null;
}

export function Upload(p: Props) {
  const [preview, setPreview] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!p.file) {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(p.file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [p.file]);

  const pick = (f: File | undefined) => {
    if (f) p.onFile(f);
  };

  return (
    <section className="card upload" aria-labelledby="upload-h">
      <h1 className="mono-label" id="upload-h">NEW SCAN</h1>
      <div
        className={`dropzone${drag ? " drag" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          if (!p.running) pick(e.dataTransfer.files[0]);
        }}
      >
        {preview ? (
          <img className="thumb" src={preview} alt="Chosen photo preview" />
        ) : (
          <UploadIcon />
        )}
        <input
          ref={inputRef}
          id="file"
          className="sr-file"
          type="file"
          accept={ACCEPT}
          disabled={p.running}
          onChange={(e) => pick(e.target.files?.[0])}
        />
        <label htmlFor="file" className="drop-label">
          <span className="drop-title">{p.file ? p.file.name : "Drop a vehicle photo here"}</span>
          <span className="muted">
            {p.file ? "Click to choose a different photo" : "or click to choose. JPEG, PNG, WebP or HEIC, up to 15 MB."}
          </span>
        </label>
      </div>

      <fieldset className="seg-wrap" disabled={p.running}>
        <legend className="mono-label">SENSITIVITY</legend>
        <div className="seg">
          {([["high_recall", "High recall"], ["standard", "Standard"]] as const).map(([v, l]) => (
            <label key={v} className="seg-opt">
              <input
                type="radio"
                name="sensitivity"
                value={v}
                checked={p.sensitivity === v}
                onChange={() => p.onSensitivity(v)}
              />
              <span>{l}</span>
            </label>
          ))}
        </div>
        <p className="muted helper">
          High recall flags more possible torn or missing parts but raises more false alarms.
          Standard is stricter.
        </p>
      </fieldset>

      <div className="actions">
        <button type="button" className="btn primary" disabled={!p.file || p.running} onClick={p.onSubmit}>
          Assess damage
        </button>
        {p.running && (
          <div className="status" role="status" aria-live="polite">
            <span className="mono">
              {p.waiting ? "Waiting for the previous scan…" : `Analysing… ${p.seconds}s`}
            </span>
            <div className="bar" aria-hidden="true"><span /></div>
          </div>
        )}
      </div>
      {p.error && <p className="error" role="alert">{p.error}</p>}
    </section>
  );
}
