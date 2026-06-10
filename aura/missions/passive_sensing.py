"""Passive on-route sensing logger: stream (t, x, v, wind) per sample."""

from __future__ import annotations

from pathlib import Path
from typing import IO

import json
import numpy as np


class PassiveSensingLogger:
    def __init__(self, out_path: str | Path):
        self.path = Path(out_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] | None = None

    def __enter__(self):
        self._fh = open(self.path, "w")
        return self

    def __exit__(self, *exc):
        if self._fh:
            self._fh.close()
            self._fh = None

    def log(self, t_s: float, uav_id: int, pos: np.ndarray, vel: np.ndarray,
            wind: np.ndarray) -> None:
        if self._fh is None:
            self._fh = open(self.path, "a")
        rec = {
            "t": float(t_s),
            "uav": int(uav_id),
            "x": [float(c) for c in pos],
            "v": [float(c) for c in vel],
            "w": [float(c) for c in wind],
        }
        self._fh.write(json.dumps(rec) + "\n")
