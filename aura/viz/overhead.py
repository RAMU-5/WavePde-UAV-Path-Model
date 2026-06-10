"""2D overhead matplotlib visualization for Stage 1.

Pattern: aura_toy_archive/main.py:Renderer ported to projected meters with
station glyphs, in-flight UAV trails, and a KPI overlay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class FrameSpec:
    fleet_pos: np.ndarray              # (N, 3)
    fleet_state: np.ndarray            # (N,) int
    station_xy: np.ndarray             # (K, 2)
    station_quotas: np.ndarray         # (K,) int
    no_fly_circles: List[tuple]        # [(cx, cy, r)]
    nominal_segments: List[np.ndarray] = field(default_factory=list)
    kpi_text: str = ""
    t_s: float = 0.0


def render_frame(ax, frame: FrameSpec) -> None:
    """Draw a single matplotlib frame onto an Axes."""
    ax.clear()
    if frame.nominal_segments:
        for seg in frame.nominal_segments:
            ax.plot(seg[:, 0], seg[:, 1], color="#cccccc", linewidth=0.4, alpha=0.6)
    for (cx, cy, r) in frame.no_fly_circles:
        circle = __import__("matplotlib.patches", fromlist=["Circle"]).Circle(
            (cx, cy), r, color="red", alpha=0.18, zorder=1,
        )
        ax.add_patch(circle)
    # Stations
    sx = frame.station_xy[:, 0]
    sy = frame.station_xy[:, 1]
    ax.scatter(sx, sy, c="black", s=24, marker="s", zorder=4)
    # UAVs colored by state
    state_colors = {0: "#888888", 1: "#1f77b4", 2: "#ff7f0e",
                    3: "#9467bd", 4: "#d62728"}
    cs = np.asarray([state_colors.get(int(s), "#888888")
                     for s in frame.fleet_state])
    ax.scatter(frame.fleet_pos[:, 0], frame.fleet_pos[:, 1],
               c=cs, s=10, zorder=5, edgecolors="none")
    ax.set_aspect("equal")
    ax.set_title(f"t = {frame.t_s:.1f} s   {frame.kpi_text}")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")


class GifRecorder:
    def __init__(self, path: str, fps: int = 20):
        self.path = path
        self.fps = fps
        self.frames: List[np.ndarray] = []

    def capture(self, fig) -> None:
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        self.frames.append(buf.copy())

    def save(self) -> None:
        if not self.frames:
            return
        try:
            import imageio.v2 as imageio
        except Exception:
            import imageio
        imageio.mimsave(self.path, self.frames, fps=self.fps)
