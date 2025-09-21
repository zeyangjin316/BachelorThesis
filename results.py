import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any


def _sanitize_for_path(s: str) -> str:
    """Make a short, filesystem-safe directory name."""
    if not s:
        return "Unknown"
    # Keep letters, numbers, hyphen/underscore/space -> convert spaces to underscores
    safe = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in str(s))
    return safe.strip().replace(" ", "_")[:40]  # keep it compact


class ResultSaver:
    """
    Directory layout (example):
    results/
      CGM/                     # experiment group (by mode)
      TWOSTEP/
        Gaussian/              # <-- grouped by copula type
          20250817-141233/     # timestamp at *completion*
            params_two_step.json
            metrics.csv
            samples_two_step.npy
            manifest.json
    """
    def __init__(self, mode: str, cgm_params: dict | None, two_step_params: dict | None):
        self.mode = mode
        self.cgm_params = cgm_params
        self.two_step_params = two_step_params
        self.copula_type = two_step_params.get("copula_type", None) if two_step_params else None

        # Defer group_path creation to save(), so we can use params if passed there
        self.group_path = None
        self.base_path = None

    # ---------- public API ----------
    def save(self, samples: dict, metrics_rows: list[dict], params: dict[str, Any] | None = None):
        """
        samples: {"samples_cgm": np.ndarray|None, "samples_two_step": np.ndarray|None}
        metrics_rows: list of dicts, one per model, containing ONLY metrics (e.g., loss, CRPS, runtime_s, etc.)
        params: {"cgm": <params dict or None>, "two_step": <params dict or None>}
        """
        params = params or {}

        # --- build group path (mode + optional copula subfolder) ---
        group_map = {"cgm": "CGM", "2step": "TWOSTEP", "comparison": "COMPARISON"}
        group = group_map.get(str(self.mode).lower(), "RUN")

        # Try two sources for copula type: constructor arg or params at save-time
        copula_type = None
        if group == "TWOSTEP" and self.copula_type is not None:
            group = os.path.join(group, _sanitize_for_path(self.copula_type))

        # Create the base group folder now
        self.group_path = os.path.join("results", group)
        os.makedirs(self.group_path, exist_ok=True)

        # completion timestamp for this run
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.base_path = os.path.join(self.group_path, ts)
        os.makedirs(self.base_path, exist_ok=True)

        # --- save params (if provided) ---
        if params.get("cgm") is not None:
            self._save_json("params_cgm.json", self._to_jsonable(params["cgm"]))
        if params.get("two_step") is not None:
            self._save_json("params_two_step.json", self._to_jsonable(params["two_step"]))

        # --- save metrics (ONLY metrics) ---
        if metrics_rows:
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
            "copula_type": copula_type if group.startswith("TWOSTEP") else None,
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
