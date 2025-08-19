import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any


class ResultSaver:
    """
    Directory layout (example):
    results/
      CGM/                     # experiment group (by mode)
        20250817-141233/       # timestamp at *completion*
          params_cgm.json      # if CGM params provided
          params_two_step.json # if Two-Step params provided
          metrics.csv          # ONLY metrics (no params)
          samples_cgm.npy      # if CGM samples exist
          samples_two_step.npy # if Two-Step samples exist
          manifest.json        # lightweight run manifest
    """
    def __init__(self, mode: str, cgm_params: dict | None, two_step_params: dict | None):
        self.mode = mode
        self.cgm_params = cgm_params
        self.two_step_params = two_step_params

        # simple, robust experiment group name based on mode
        group = {"cgm": "CGM", "2step": "TWOSTEP", "comparison": "COMPARISON"}.get(mode, "RUN")

        # create the base folder now; timestamp is added on save() at completion time
        self.group_path = os.path.join("results", group)
        os.makedirs(self.group_path, exist_ok=True)

        # this will be set on first save() call using current completion timestamp
        self.base_path = None

    # ---------- public API ----------
    def save(self, samples: dict, metrics_rows: list[dict], params: dict[str, Any] | None = None):
        """
        samples: {"samples_cgm": np.ndarray|None, "samples_two_step": np.ndarray|None}
        metrics_rows: list of dicts, one per model, containing ONLY metrics (e.g., loss, CRPS, runtime_s, etc.)
        params: {"cgm": <params dict or None>, "two_step": <params dict or None>}
        """
        # completion timestamp for this run
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.base_path = os.path.join(self.group_path, ts)
        os.makedirs(self.base_path, exist_ok=True)

        # --- save params (if provided) ---
        params = params or {}
        if params.get("cgm") is not None:
            self._save_json("params_cgm.json", self._to_jsonable(params["cgm"]))
        if params.get("two_step") is not None:
            self._save_json("params_two_step.json", self._to_jsonable(params["two_step"]))

        # --- save metrics (ONLY metrics) ---
        if metrics_rows:
            # Ensure no accidental nested configs: they should be metrics only
            cleaned = [self._strip_non_jsonable(row) for row in metrics_rows]
            df_metrics = pd.DataFrame(cleaned)
            df_metrics.to_csv(os.path.join(self.base_path, "metrics.csv"), index=False)

        # --- save samples (if exist) ---
        samples_cgm = (samples or {}).get("samples_cgm", None)
        samples_two_step = (samples or {}).get("samples_two_step", None)

        if samples_cgm is not None:
            np.save(os.path.join(self.base_path, "samples_cgm.npy"), samples_cgm)
        if samples_two_step is not None:
            np.save(os.path.join(self.base_path, "samples_two_step.npy"), samples_two_step)

        # --- save a tiny manifest for convenience ---
        manifest = {
            "mode": self.mode,
            "completed_at": ts,
            "paths": {
                "params_cgm": "params_cgm.json" if params.get("cgm") is not None else None,
                "params_two_step": "params_two_step.json" if params.get("two_step") is not None else None,
                "metrics": "metrics.csv" if metrics_rows else None,
                "samples_cgm": "samples_cgm.npy" if samples_cgm is not None else None,
                "samples_two_step": "samples_two_step.npy" if samples_two_step is not None else None,
            },
        }
        self._save_json("manifest.json", manifest)

        print(f"✅ Results saved to: {self.base_path}")

    # ---------- helpers ----------
    def _save_json(self, filename: str, payload: Any):
        with open(os.path.join(self.base_path, filename), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _to_jsonable(self, obj: Any) -> Any:
        """
        Recursively convert objects (including dataclass-like configs) into JSONable structures.
        Falls back to string for anything stubborn.
        """
        # primitives
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj

        # numpy types
        try:
            import numpy as _np
            if isinstance(obj, (_np.generic,)):
                return obj.item()
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
        except Exception:
            pass

        # dict-like
        if isinstance(obj, dict):
            return {str(k): self._to_jsonable(v) for k, v in obj.items()}

        # list/tuple/set
        if isinstance(obj, (list, tuple, set)):
            return [self._to_jsonable(v) for v in obj]

        # dataclass or simple objects
        # try dataclasses.asdict first
        try:
            from dataclasses import is_dataclass, asdict
            if is_dataclass(obj):
                return self._to_jsonable(asdict(obj))
        except Exception:
            pass

        # try __dict__
        if hasattr(obj, "__dict__") and obj.__dict__:
            return self._to_jsonable(obj.__dict__)

        # fallback: string representation
        return str(obj)

    def _strip_non_jsonable(self, row: dict) -> dict:
        """Ensure metrics rows are JSON/CSV friendly (e.g., cast numpy scalars)."""
        return {k: self._to_jsonable(v) for k, v in row.items()}
