"""Population statistics tracker for Evolutionary Ecosystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .sim import World


@dataclass
class PopulationStats:
    tick: int = 0
    population: int = 0
    avg_generation: float = 0.0
    max_generation: int = 0
    total_births: int = 0
    total_deaths: int = 0
    avg_stats: dict[str, float] = field(default_factory=dict)
    avg_behavior: dict[str, float] = field(default_factory=dict)
    epoch: str = ""
    weather: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "population": self.population,
            "avg_generation": round(self.avg_generation, 1),
            "max_generation": self.max_generation,
            "total_births": self.total_births,
            "total_deaths": self.total_deaths,
            "avg_stats": {k: round(v, 3) for k, v in self.avg_stats.items()},
            "avg_behavior": {k: round(v, 3) for k, v in self.avg_behavior.items()},
            "epoch": self.epoch,
            "weather": self.weather,
        }


class PopulationTracker:
    def __init__(self, history_length: int = 500):
        self.history: list[PopulationStats] = []
        self._max_history = history_length

    def update(self, world: "World") -> PopulationStats:
        """Compute averages across living creatures, append to history."""
        from .gene_engine import SITUATION_NAMES

        creatures = list(world.creatures.values())
        n = len(creatures)

        generations = [c.generation for c in creatures]
        avg_gen = sum(generations) / n if n else 0.0
        max_gen = max(generations) if generations else 0

        # Average EntityStats
        avg_stats: dict[str, float] = {}
        if n > 0:
            stat_sums: dict[str, float] = {}
            for c in creatures:
                if c.stats:
                    for k, v in c.stats.to_dict().items():
                        stat_sums[k] = stat_sums.get(k, 0.0) + v
            avg_stats = {k: v / n for k, v in stat_sums.items()}

        # Average BehaviorProfile strengths
        avg_behavior: dict[str, float] = {}
        behavior_creatures = [c for c in creatures if c.behavior_profile is not None]
        if behavior_creatures:
            for sit_name in SITUATION_NAMES:
                total = sum(c.behavior_profile.strength(sit_name) for c in behavior_creatures)
                avg_behavior[sit_name] = total / len(behavior_creatures)

        stats = PopulationStats(
            tick=world.tick_count,
            population=n,
            avg_generation=avg_gen,
            max_generation=max_gen,
            total_births=world.total_births,
            total_deaths=world.total_deaths,
            avg_stats=avg_stats,
            avg_behavior=avg_behavior,
            epoch=world.epoch.name,
            weather=world.weather,
        )

        self.history.append(stats)
        if len(self.history) > self._max_history:
            self.history = self.history[-self._max_history:]

        return stats
