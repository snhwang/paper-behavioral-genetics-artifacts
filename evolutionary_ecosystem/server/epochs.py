"""Epoch and Weather system for Evolutionary Ecosystem.

Epochs cycle through different environmental pressures that modulate
both the fast path (food availability, weather damage, combat probability)
and the slow path (BEAR context tags shift which instructions are retrieved).
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Epoch definitions
# ---------------------------------------------------------------------------


@dataclass
class Epoch:
    name: str
    food_multiplier: float    # affects food spawn rate
    weather_severity: float   # affects weather damage
    aggression_bonus: float   # added to combat probability checks
    description: str = ""


EPOCHS = [
    Epoch("abundance", 1.5, 0.2, 0.0,  "Plentiful food, mild weather"),
    Epoch("ice_age",   0.7, 1.0, 0.1,  "Harsh cold, reduced food"),
    Epoch("predator_bloom", 1.0, 0.4, 0.2, "Increased aggression across population"),
    Epoch("expansion", 1.3, 0.3, 0.0,  "Good conditions for exploration and breeding"),
    Epoch("famine",    0.5, 0.8, 0.1,  "Food shortage"),
]

# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

WEATHER_TYPES = ["mild", "storm", "heat", "cold"]

# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------

EPOCH_DURATION_TICKS = 3000
WEATHER_CHANGE_INTERVAL = 500
WEATHER_DAMAGE = 0.2
