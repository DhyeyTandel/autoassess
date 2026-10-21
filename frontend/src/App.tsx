import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, assess, health } from "./api/client";
import { fixtureResult } from "./api/fixture";
import type { AssessResult, Sensitivity } from "./api/types";
import { MoonIcon, SunIcon } from "./components/Icons";
import { Result } from "./components/Result";
import { Upload } from "./components/Upload";
import { formatSeconds } from "./lib/format";
import { initialTheme, prefersDark, readStoredTheme, storeTheme, type Theme } from "./lib/theme";

const useFixture = import.meta.env.DEV && new URLSearchParams(window.location.search).has("fixture");

export function App() {
  const [theme, setTheme] = useState<Theme>(() => initialTheme(readStoredTheme(), prefersDark()));
  const [file, setFile] = useState<File | null>(null);
  const [sensitivity, setSensitivity] = useState<Sensitivity>("high_recall");
  const [running, setRunning] = useState(false);
  const [waiting, setWaiting] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AssessResult | null>(useFixture ? fixtureResult : null);
  const [resultName, setResultName] = useState(useFixture ? "fixture.jpg" : "");
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  useEffect(() => () => window.clearInterval(timer.current), []);

  const toggle = () => {
    const next: Theme = theme === "dark" ? "light" : "dark";
    setTheme(next);
    storeTheme(next);
  };

  const submit = useCallback(async () => {
    if (!file) return;
    setError(null);
    setRunning(true);
    setSeconds(0);
    const t0 = Date.now();
    window.clearInterval(timer.current);
    timer.current = window.setInterval(() => setSeconds(Math.floor((Date.now() - t0) / 1000)), 500);
    try {
      const h = await health();
      setWaiting(h.busy);
      setResult(await assess(file, sensitivity));
      setResultName(file.name);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong. Try again.");
    } finally {
      window.clearInterval(timer.current);
      setRunning(false);
      setWaiting(false);
    }
  }, [file, sensitivity]);

  const reset = () => {
    setResult(null);
    setFile(null);
    setError(null);
  };

  const dark = theme === "dark";
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="mark" aria-hidden="true" />
          <span className="brand-name">AUTOASSESS</span>
          <span className="mono muted meta">
            {result ? `CPU · ${formatSeconds(result.elapsed_ms)}s` : "Research demo"}
          </span>
        </div>
        <div className="top-actions">
          <button
            type="button"
            className="btn icon"
            onClick={toggle}
            aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
          >
            {dark ? <SunIcon /> : <MoonIcon />}
          </button>
          {result && (
            <button type="button" className="btn primary" onClick={reset}>
              Scan another photo
            </button>
          )}
        </div>
      </header>
      <main>
        {result ? (
          <Result r={result} filename={resultName} />
        ) : (
          <Upload
            file={file}
            onFile={(f) => { setFile(f); setError(null); }}
            sensitivity={sensitivity}
            onSensitivity={setSensitivity}
            onSubmit={submit}
            running={running}
            waiting={waiting}
            seconds={seconds}
            error={error}
          />
        )}
      </main>
      <footer className="notice">
        Research demo, non-commercial. Models trained on CarDD, VehiDE and Carparts-Seg. Photos
        are processed and not stored.
      </footer>
    </div>
  );
}
