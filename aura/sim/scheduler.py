"""Layered fixed-timestep scheduler.

A single SimClock advances in tick_s steps. Each layer registers a frequency
and a callback `cb(now_s, dt_layer)`. Higher-frequency layers fire multiple
times per slow-layer tick. Callbacks are pure functions of the world state
(`SimState`); ordering follows the registration order so 100 Hz physics
runs before 50 Hz MPC, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List


LayerCb = Callable[[float, float], None]


@dataclass
class Layer:
    name: str
    rate_hz: float
    callback: LayerCb
    next_fire_at: float = 0.0


@dataclass
class SimClock:
    tick_s: float = 0.01
    now_s: float = 0.0
    layers: List[Layer] = field(default_factory=list)

    def add_layer(self, name: str, rate_hz: float, callback: LayerCb) -> None:
        self.layers.append(Layer(name=name, rate_hz=rate_hz, callback=callback,
                                 next_fire_at=0.0))

    def step(self) -> None:
        """Advance one base tick. Fire any layer whose next_fire_at <= now."""
        for layer in self.layers:
            while layer.next_fire_at <= self.now_s + 1e-12:
                period = 1.0 / layer.rate_hz
                layer.callback(self.now_s, period)
                layer.next_fire_at += period
        self.now_s += self.tick_s

    def run(self, duration_s: float) -> None:
        n_ticks = int(round(duration_s / self.tick_s))
        for _ in range(n_ticks):
            self.step()


def run(clock: SimClock, duration_s: float) -> None:
    clock.run(duration_s)
