"""Simulation physics and world state for Evolutionary Ecosystem.

Merges creature ecosystem's NPC physics (movement, skill animations, breeding,
happiness, aging, rage, predator raids, items) with spatial evolution's
survival mechanics (HP, energy, metabolism, food spawning, weather, epochs,
fast-path BEAR-driven movement).
"""

from __future__ import annotations

import logging
import math
import random
import time

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from typing import Any

from .gene_engine import (
    AppearanceParams, SkillSet, EntityStats, BehaviorProfile, SITUATION_NAMES,
)
from .epochs import (
    Epoch, EPOCHS,
    EPOCH_DURATION_TICKS, WEATHER_CHANGE_INTERVAL, WEATHER_DAMAGE,
    WEATHER_TYPES,
)

# ---------------------------------------------------------------------------
# World constants
# ---------------------------------------------------------------------------

WORLD_W = 20.0
WORLD_H = 14.0
PHYSICS_HZ = 10

SPEED_MAP = {
    "slow":   0.5,
    "normal": 1.0,
    "fast":   2.0,
    "sprint": 3.5,
}

VALID_ANIMATIONS = {
    "idle", "walk", "roll", "circle", "bounce", "sneak",
    "sprint", "excited", "nuzzle", "approach", "flee", "rally",
}

VALID_MOODS = {"neutral", "happy", "playful", "cautious", "excited", "sleepy", "annoyed"}

VALID_EFFECTS = {"hearts", "sparkles", "question_mark", "exclamation", "zzz", "sweat", "shield"}

BREED_DISTANCE  = 2.5
BREED_HAPPINESS = 45.0
BREED_COOLDOWN  = 30.0

# Aging
MAX_AGE_MIN    = 300.0
MAX_AGE_MAX    = 500.0

MAX_POPULATION = 50   # safety cap; environment should self-regulate below this

# Rage / rabid mechanic
RAGE_THRESHOLD     = 80.0
RAGE_BUILDUP_RATE  = 3.0
RAGE_DECAY_RATE    = 5.0
CHALLENGE_DAMAGE   = 12.0
DEFENSE_DAMAGE     = 20.0
DEFENSE_RANGE      = 3.5
DEFENSE_QUORUM     = 2

# Predator raid mechanic
PREDATOR_SPAWN_INTERVAL  = 150.0
PREDATOR_SPEED           = 2.0
PREDATOR_ATTACK_RANGE    = 1.2
PREDATOR_ATTACK_DAMAGE   = 10.0
PREDATOR_ATTACK_COOLDOWN = 5.0
PREDATOR_DRIVE_OFF_COUNT = 3
PREDATOR_DRIVE_OFF_RANGE = 3.0
PREDATOR_DETECT_RANGE    = 8.0
PREDATOR_PANIC_RANGE     = 2.5

# Energy / metabolism (from spatial evolution)
ENERGY_MAX       = 100.0
BASE_METABOLISM  = 0.012  # energy drain per tick
STARVATION_DMG   = 0.4    # HP lost per tick when energy <= 0
FOOD_ENERGY      = 30.0
FOOD_PICKUP_RANGE = 1.0

# Food spawning
FOOD_SPAWN_BASE = 0.25    # probability per tick
MAX_FOOD = 50

# Combat
BASE_DAMAGE    = 15.0
DEFENSE_FACTOR = 0.6

# Item constants
ITEM_TYPES = {"flower", "tree", "rock", "food", "ball"}
FOOD_EAT_DISTANCE    = 0.8
FOOD_HAPPINESS_BOOST = 20.0
FOOD_EAT_MIN_HAPPY   = 80.0
FOOD_RESPAWN_TIME    = 45.0

BALL_PUSH_DIST     = 0.8
BALL_PUSH_STRENGTH = 4.0
BALL_FRICTION      = 0.4

FLOWER_HAPPY_RATE  = 0.06
FLOWER_RANGE       = 1.2

# ---------------------------------------------------------------------------
# Food dataclass (for ecosystem food spawning)
# ---------------------------------------------------------------------------


@dataclass
class FoodItem:
    x: float
    y: float
    energy: float = FOOD_ENERGY

    def to_dict(self) -> dict[str, Any]:
        return {"x": round(self.x, 2), "y": round(self.y, 2)}


# ---------------------------------------------------------------------------
# WorldItem dataclass
# ---------------------------------------------------------------------------


@dataclass
class WorldItem:
    id:         str
    type:       str
    x:          float
    y:          float
    vx:         float = 0.0
    vy:         float = 0.0
    active:     bool  = True
    respawn_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id":     self.id,
            "type":   self.type,
            "x":      round(self.x, 3),
            "y":      round(self.y, 3),
            "vx":     round(self.vx, 3),
            "vy":     round(self.vy, 3),
            "active": self.active,
        }


# ---------------------------------------------------------------------------
# Predator dataclass
# ---------------------------------------------------------------------------


@dataclass
class Predator:
    x:             float
    y:             float
    vx:            float = 0.0
    vy:            float = 0.0
    active:        bool  = False
    heading:       float = 0.0
    target_id:     str | None = None
    attack_cd:     float = 0.0
    driven_off:    bool  = False
    spawn_at:      float = 0.0
    kills:         int   = 0
    max_kills:     int   = 2    # leaves after this many kills

    def to_dict(self) -> dict[str, Any]:
        return {
            "x":         round(self.x, 3),
            "y":         round(self.y, 3),
            "active":    self.active,
            "heading":   round(self.heading, 3),
            "target_id": self.target_id,
        }


# ---------------------------------------------------------------------------
# Creature dataclass (merged creature + spatial fields)
# ---------------------------------------------------------------------------


@dataclass
class Creature:
    id:         str
    name:       str
    x:          float
    y:          float
    z:          float = 0.0
    vx:         float = 0.0
    vy:         float = 0.0

    # Gene-derived traits
    genes:      dict[str, str]           = field(default_factory=dict)
    appearance: AppearanceParams         = field(default_factory=AppearanceParams)
    skills:     SkillSet                 = field(default_factory=SkillSet)
    behavior_profile: BehaviorProfile | None = None

    # EntityStats (from spatial evolution)
    stats:      EntityStats | None       = None

    # State
    happiness:  float  = 70.0
    mood:       str    = "neutral"
    animation:  str    = "idle"
    effect:     str | None = None
    thought:    str | None = None
    intent:     str    = "wander"

    # HP and Energy (from spatial evolution)
    hp:         float  = 100.0
    energy:     float  = 100.0

    # Breeding
    breed_cooldown: float = 0.0
    generation:     int   = 0
    parents:        tuple[str, str] | None = None
    children_count: int   = 0

    # Aging / lifespan
    age:     float = 0.0
    max_age: float = 120.0
    fading:  bool  = False

    # Conflict / vitality
    vitality:  float = 100.0
    rage:      float = 0.0
    is_rabid:  bool  = False
    kills:     int   = 0

    # Movement
    target:            dict | None = None
    circle_waypoints:  list[dict]  = field(default_factory=list)
    circle_idx:        int         = 0
    busy_until:        float       = 0.0
    heading:           float       = 0.0

    # Social / conversation state
    speak_to:       str | None = None
    last_speech:    str | None = None
    last_retrieval: dict | None = None

    # Brain bookkeeping
    corpus:     Any = field(default=None, repr=False)
    retriever:  Any = field(default=None, repr=False)  # Retriever | None

    def is_busy(self, t: float) -> bool:
        return self.busy_until > t

    def bear_strength(self, query: str, tags: list | None = None,
                      fallback: float = 0.3) -> float:
        """Retrieve behavior strength from corpus via BEAR. Falls back to scalar.
        Results are cached per query — corpus never changes after birth."""
        cache_key = (query, tuple(tags or []))
        if hasattr(self, "_bear_cache") and cache_key in self._bear_cache:
            return self._bear_cache[cache_key]
        if self.retriever is None:
            result = (self.behavior_profile.strength(query.split(",")[0].strip())
                      if self.behavior_profile else fallback)
        else:
            from bear import Context
            ctx = Context(tags=tags or [])
            try:
                results = self.retriever.retrieve(query=query, context=ctx,
                                                  top_k=1, threshold=0.0)
                result = results[0].similarity if results else fallback
            except Exception:
                result = fallback
        if not hasattr(self, "_bear_cache"):
            self._bear_cache = {}
        self._bear_cache[cache_key] = result
        return result

    def can_breed(self, breed_happiness: float = BREED_HAPPINESS) -> bool:
        return (
            self.breed_cooldown <= 0
            and self.happiness >= breed_happiness
            and self.energy >= 30.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id":         self.id,
            "name":       self.name,
            "x":          round(self.x, 3),
            "y":          round(self.y, 3),
            "z":          round(self.z, 3),
            "vx":         round(self.vx, 3),
            "vy":         round(self.vy, 3),
            "appearance": self.appearance.to_dict(),
            "skills":     self.skills.to_dict(),
            "happiness":  round(self.happiness, 1),
            "mood":       self.mood,
            "animation":  self.animation,
            "effect":     self.effect,
            "thought":    self.thought,
            "intent":     self.intent,
            "generation": self.generation,
            "parents":    list(self.parents) if self.parents else None,
            "heading":    round(self.heading, 3),
            "hp":         round(self.hp, 1),
            "energy":     round(self.energy, 1),
            "breed_ready": self.can_breed(),
            "speak_to":      self.speak_to,
            "last_speech":   self.last_speech,
            "last_retrieval": self.last_retrieval,
            "age":      round(self.age, 1),
            "max_age":  round(self.max_age, 1),
            "fading":   self.fading,
            "vitality": round(self.vitality, 1),
            "rage":     round(self.rage, 1),
            "is_rabid": self.is_rabid,
            "kills":    self.kills,
            "children_count": self.children_count,
            "behavior": self.behavior_profile.to_dict() if self.behavior_profile else None,
            "stats":    self.stats.to_dict() if self.stats else None,
        }


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------


@dataclass
class World:
    creatures:     dict[str, Creature]   = field(default_factory=dict)
    items:         dict[str, WorldItem]  = field(default_factory=dict)
    food:          list[FoodItem]        = field(default_factory=list)
    breed_queue:   "asyncio.Queue | None" = field(default=None, repr=False)
    birth_queue:   "asyncio.Queue | None" = field(default=None, repr=False)
    paused:        bool  = False
    speed_mult:    float = 1.0
    bear_disabled: bool  = False
    autonomous_breeding: bool = False
    t:             float = field(default_factory=time.time)
    _id_counter:   int   = 0
    retired_ids:   list  = field(default_factory=list)
    death_log_pending: list = field(default_factory=list)  # flushed each tick by app.py
    challenge_queue: list = field(default_factory=list)
    predator:      "Predator | None" = field(default=None)

    # Epoch / weather (from spatial evolution)
    epoch:         Epoch = field(default_factory=lambda: EPOCHS[0])
    epoch_index:   int   = 0
    epoch_timer:   int   = 0
    weather:       str   = "mild"
    weather_timer: int   = 0
    tick_count:    int   = 0

    # Population stats
    total_births:  int   = 0
    total_deaths:  int   = 0

    # Breeding mode
    recombination: str = "locus"       # "locus" | "blend" | "splice"
    ploidy:        str = "haploid"     # "haploid" | "diploid_dominant" | "diploid_codominant"

    # Population auto-regulation
    auto_regulate:      bool  = False
    auto_regulate_target: int = 20

    # Environment amplifiers (UI-adjustable multipliers on gene-driven base values)
    amplifiers: dict = field(default_factory=lambda: {
        "food": 1.0,
        "metabolism": 1.0,
        "predator": 1.0,
        "aggression": 1.0,
        "breeding": 1.0,
        "weather": 1.0,
        "herding": 1.0,
    })

    # Tunable simulation parameters (adjustable via UI)
    sim_params: dict = field(default_factory=lambda: {
        "breed_distance": BREED_DISTANCE,
        "breed_cooldown": BREED_COOLDOWN,
        "breed_happiness": BREED_HAPPINESS,
        "food_spawn_rate": FOOD_SPAWN_BASE,
        "max_food": MAX_FOOD,
        "predator_interval": PREDATOR_SPAWN_INTERVAL,
        "predator_damage": PREDATOR_ATTACK_DAMAGE,
        "metabolism": BASE_METABOLISM,
    })

    def next_id(self) -> str:
        self._id_counter += 1
        return f"c{self._id_counter}"

    def get_state(self) -> dict[str, Any]:
        return {
            "type":          "state",
            "t":             round(self.t, 3),
            "tick":          self.tick_count,
            "paused":        self.paused,
            "speed_mult":    self.speed_mult,
            "bear_disabled": self.bear_disabled,
            "creatures":     [c.to_dict() for c in self.creatures.values()],
            "items":         [i.to_dict() for i in self.items.values()],
            "food":          [f.to_dict() for f in self.food],
            "predator":      self.predator.to_dict() if self.predator else None,
            "epoch":         self.epoch.name,
            "epoch_desc":    self.epoch.description,
            "weather":       self.weather,
            "total_births":  self.total_births,
            "total_deaths":  self.total_deaths,
            "recombination": self.recombination,
            "ploidy":        self.ploidy,
        }


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------


def clamp_pos(x: float, y: float) -> tuple[float, float]:
    margin = 0.5
    return (
        max(margin, min(WORLD_W - margin, x)),
        max(margin, min(WORLD_H - margin, y)),
    )


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(bx - ax, by - ay)


def _step_toward(c: Creature, tx: float, ty: float, speed: float, dt: float) -> None:
    dx, dy = tx - c.x, ty - c.y
    dist = math.hypot(dx, dy)
    if dist < 0.05:
        return
    step = speed * dt
    if step >= dist:
        c.x, c.y = tx, ty
        c.vx, c.vy = 0.0, 0.0
    else:
        ratio = step / dist
        c.x += dx * ratio
        c.y += dy * ratio
        c.vx = dx / dist * speed
        c.vy = dy / dist * speed
        c.heading = math.atan2(dy, dx)

    c.x, c.y = clamp_pos(c.x, c.y)


def _direction_to(ax: float, ay: float, bx: float, by: float) -> tuple[float, float]:
    dx, dy = bx - ax, by - ay
    dist = math.hypot(dx, dy)
    if dist < 0.01:
        return 0.0, 0.0
    return dx / dist, dy / dist


# ---------------------------------------------------------------------------
# Simulation tick
# ---------------------------------------------------------------------------

_HAPPINESS_DECAY = 0.5
_MOOD_THRESHOLDS = [
    (85, "happy"),
    (65, "neutral"),
    (45, "cautious"),
    (20, "sleepy"),
    (0,  "annoyed"),
]


def _update_mood(c: Creature) -> None:
    for threshold, mood in _MOOD_THRESHOLDS:
        if c.happiness >= threshold:
            c.mood = mood
            return
    c.mood = "annoyed"


def tick(world: World, rng: random.Random, dt: float) -> None:
    """Advance simulation by dt seconds."""
    world.t += dt
    world.tick_count += 1

    for c in list(world.creatures.values()):
        # Aging
        c.age += dt
        fade_start = c.max_age * 0.80
        if c.age >= c.max_age:
            logger.info("DEATH [age] %s (gen %d) lived %.0fs", c.name, c.generation, c.age)
            world.death_log_pending.append({
                "tick": world.tick_count, "epoch": world.epoch.name,
                "name": c.name, "generation": c.generation,
                "age": round(c.age, 1), "cause": "age",
                "children": c.children_count,
                "genes": dict(c.genes) if hasattr(c, 'genes') else {},
            })
            world.retired_ids.append(c.id)
            del world.creatures[c.id]
            world.total_deaths += 1
            continue
        c.fading = c.age >= fade_start

        # HP death
        if c.hp <= 0 or c.vitality <= 0:
            cause = "starvation" if c.energy <= 0 else "vitality" if c.vitality <= 0 else "hp"
            logger.info("DEATH [%s] %s (gen %d) age=%.0fs hp=%.0f vit=%.0f energy=%.0f",
                        cause, c.name, c.generation, c.age, c.hp, c.vitality, c.energy)
            world.death_log_pending.append({
                "tick": world.tick_count, "epoch": world.epoch.name,
                "name": c.name, "generation": c.generation,
                "age": round(c.age, 1), "cause": cause,
                "children": c.children_count,
                "genes": dict(c.genes) if hasattr(c, 'genes') else {},
            })
            world.retired_ids.append(c.id)
            del world.creatures[c.id]
            world.total_deaths += 1
            continue

        # Rage tick
        near_count = sum(
            1 for o in world.creatures.values()
            if o.id != c.id and distance(c.x, c.y, o.x, o.y) < 3.0
        )
        # Rage buildup scales with combat gene — aggressive creatures rage faster
        combat_str = c.bear_strength("fight aggression territorial combat", ["combat", "nearby"])
        aggr_amp = world.amplifiers["aggression"]
        if c.happiness < 30 and near_count >= 1:
            c.rage = min(100.0, c.rage + RAGE_BUILDUP_RATE * combat_str * aggr_amp * dt)
        else:
            c.rage = max(0.0, c.rage - RAGE_DECAY_RATE * dt)

        # Random rage spikes — only for aggressive creatures
        if combat_str > 0.38 and rng.random() < 0.0005 * combat_str * aggr_amp * dt:
            c.rage = min(100.0, c.rage + rng.uniform(10.0, 20.0) * combat_str)

        c.is_rabid = c.rage >= RAGE_THRESHOLD

        # Countdown timers
        if c.breed_cooldown > 0:
            c.breed_cooldown = max(0.0, c.breed_cooldown - dt)

        # Happiness decay toward 50
        if c.happiness > 50:
            c.happiness = max(50.0, c.happiness - _HAPPINESS_DECAY * dt)
        elif c.happiness < 50:
            c.happiness = min(50.0, c.happiness + _HAPPINESS_DECAY * 0.3 * dt)
        c.happiness = max(0.0, min(100.0, c.happiness))

        _update_mood(c)

        # Proximity happiness bonus
        for other in world.creatures.values():
            if other.id == c.id:
                continue
            d = distance(c.x, c.y, other.x, other.y)
            if d < 2.5:
                c.happiness = min(100.0, c.happiness + 0.3 * dt)

        # Movement (fast path or target-following)
        _move_creature(c, world, rng, dt)

    # New tick phases from spatial evolution
    _metabolism(world)
    _spawn_food(world, rng)
    _food_pickup(world)
    _weather_tick(world, rng)
    _epoch_check(world)

    # Process challenges + collective defense
    _process_challenges(world)

    # Auto-regulate population pressure
    _auto_regulate_population(world)

    # BEAR triggers breeding via [!breed(nearest)]; autonomous fallback
    if world.autonomous_breeding:
        _check_breeding(world, rng)

    # Predator raid
    _tick_predator(world, rng, dt)

    # Item physics and creature-item interactions
    _tick_items(world, dt)

    # Social mechanics
    _tick_social(world)


# ---------------------------------------------------------------------------
# New tick phases (from spatial evolution)
# ---------------------------------------------------------------------------


def _metabolism(world: World) -> None:
    """Drain energy per tick, starvation damage when energy <= 0.

    Energy drain scales inversely with food_seeking strength (efficient
    foragers waste less energy).  Starvation HP damage scales inversely
    with survival strength (resilient creatures resist starvation longer).
    """
    for c in world.creatures.values():
        speed_factor = c.stats.speed if c.stats else 0.5
        # Behavior-dependent efficiency: strong food_seeking → lower drain
        food_str = c.bear_strength("hungry foraging find food eat", ["food", "hunger"])
        efficiency = 1.15 - food_str  # range ~[0.9, 1.1] for str in [0.05, 0.25]
        base_meta = world.sim_params.get("metabolism", BASE_METABOLISM)
        drain = (base_meta + speed_factor * 0.01) * efficiency * world.amplifiers["metabolism"]
        c.energy -= drain
        if c.energy <= 0:
            c.energy = 0
            # Behavior-dependent starvation resistance
            surv_str = c.bear_strength("survive starvation endurance resilience", ["survival", "hunger"])
            starve_factor = 1.4 - surv_str  # strong survival → less damage
            c.hp -= STARVATION_DMG * starve_factor * world.amplifiers["metabolism"] * (1.0 / PHYSICS_HZ)


def _spawn_food(world: World, rng: random.Random) -> None:
    """Randomly spawn food based on epoch."""
    max_food = int(world.sim_params["max_food"])
    if len(world.food) >= max_food:
        return
    rate = world.sim_params["food_spawn_rate"] * world.epoch.food_multiplier * world.amplifiers["food"]
    if rng.random() < rate:
        world.food.append(FoodItem(
            x=rng.uniform(1.0, WORLD_W - 1.0),
            y=rng.uniform(1.0, WORLD_H - 1.0),
        ))


def _food_pickup(world: World) -> None:
    """Creatures near food collect it.

    Energy gained scales with food_seeking strength (better foragers
    extract more nutrition from each food item).
    """
    eaten: list[int] = []
    for i, food in enumerate(world.food):
        for c in world.creatures.values():
            d = distance(c.x, c.y, food.x, food.y)
            pickup_r = FOOD_PICKUP_RANGE * (0.5 + (c.stats.food_finding if c.stats else 0.5))
            if d < pickup_r:
                food_str = c.bear_strength("hungry foraging find food eat", ["food", "hunger"])
                gain = food.energy * (0.8 + food_str * 2.0)  # rescaled for new range
                c.energy = min(ENERGY_MAX, c.energy + gain)
                c.happiness = min(100.0, c.happiness + 5.0)
                eaten.append(i)
                break
    for i in sorted(eaten, reverse=True):
        world.food.pop(i)


def _weather_tick(world: World, rng: random.Random) -> None:
    """Apply weather effects and cycle weather types."""
    world.weather_timer -= 1
    if world.weather_timer <= 0:
        world.weather_timer = WEATHER_CHANGE_INTERVAL
        severity = world.epoch.weather_severity
        if rng.random() < severity:
            world.weather = rng.choice(["storm", "heat", "cold"])
        else:
            world.weather = "mild"

    if world.weather != "mild":
        for c in world.creatures.values():
            resist = c.stats.climate_resist if c.stats else 0.3
            dmg = WEATHER_DAMAGE * (1 - resist * 0.8) * world.amplifiers["weather"] * (1.0 / PHYSICS_HZ)
            dmg = max(0.01, dmg)
            c.hp -= dmg


def _epoch_check(world: World) -> None:
    """Advance epoch after EPOCH_DURATION_TICKS."""
    if getattr(world, 'epoch_locked', False):
        return  # epoch locked for eval mode
    world.epoch_timer += 1
    if world.epoch_timer >= EPOCH_DURATION_TICKS:
        world.epoch_timer = 0
        world.epoch_index = (world.epoch_index + 1) % len(EPOCHS)
        world.epoch = EPOCHS[world.epoch_index]


# ---------------------------------------------------------------------------
# Fast-path movement
# ---------------------------------------------------------------------------


    # (_fast_path_move removed — all movement is now BEAR-driven
    # through retrieved gene templates with embedded action markers)


# ---------------------------------------------------------------------------
# Movement dispatch
# ---------------------------------------------------------------------------


def _move_creature(c: Creature, world: World, rng: random.Random, dt: float) -> None:
    """Apply movement based on current target and skills, with fast-path fallback."""
    # Determine effective speed
    base_speed = c.skills.speed_base
    if c.animation == "sprint":
        base_speed *= 2.0
    elif c.animation == "sneak":
        base_speed *= 0.4
    elif c.animation == "roll":
        base_speed *= 1.2

    # Circle movement
    if c.animation == "circle" and c.circle_waypoints:
        wp = c.circle_waypoints[c.circle_idx % len(c.circle_waypoints)]
        _step_toward(c, wp["x"], wp["y"], base_speed, dt)
        if distance(c.x, c.y, wp["x"], wp["y"]) < 0.15:
            c.circle_idx = (c.circle_idx + 1) % len(c.circle_waypoints)
        return

    # LLM target override — slow path sets target, fast path defers
    if c.target:
        tx, ty = c.target["x"], c.target["y"]
        _step_toward(c, tx, ty, base_speed, dt)
        if distance(c.x, c.y, tx, ty) < 0.15:
            c.target = None
            c.vx, c.vy = 0.0, 0.0
            if c.animation in ("walk", "sprint", "sneak"):
                c.animation = "walk"
            # Fall through to fast-path instead of standing still
        else:
            return

    # No target — creature waits for next BEAR brain tick to set a new one.
    # All movement is driven by BEAR retrieval + embedded action markers.
    c.vx, c.vy = 0.0, 0.0


# ---------------------------------------------------------------------------
# Challenges, breeding, predator, items, social
# ---------------------------------------------------------------------------


def _process_challenges(world: World) -> None:
    if not world.challenge_queue:
        return

    for atk_id, tgt_id in world.challenge_queue:
        attacker = world.creatures.get(atk_id)
        target   = world.creatures.get(tgt_id)
        if not attacker or not target:
            continue

        target.vitality  = max(0.0, target.vitality - CHALLENGE_DAMAGE)
        target.happiness = max(0.0, target.happiness - 10.0)
        target.effect    = "sweat"

        defenders = [
            o for o in world.creatures.values()
            if o.id != atk_id
            and not o.is_rabid
            and distance(attacker.x, attacker.y, o.x, o.y) <= DEFENSE_RANGE
        ]
        if len(defenders) >= DEFENSE_QUORUM:
            attacker.vitality = max(0.0, attacker.vitality - DEFENSE_DAMAGE)
            attacker.rage     = max(0.0, attacker.rage - 35.0)
            attacker.happiness = max(0.0, attacker.happiness - 15.0)
            for d in defenders[:DEFENSE_QUORUM]:
                d.effect    = "shield"
                d.happiness = min(100.0, d.happiness + 5.0)

    world.challenge_queue.clear()


def _tick_predator(world: World, rng: random.Random, dt: float) -> None:
    p = world.predator
    if p is None:
        return

    if not p.active:
        if world.t >= p.spawn_at:
            if world.creatures:
                edge = rng.choice(["top", "bottom", "left", "right"])
                if edge == "top":
                    p.x, p.y = rng.uniform(1.0, WORLD_W - 1.0), 0.0
                elif edge == "bottom":
                    p.x, p.y = rng.uniform(1.0, WORLD_W - 1.0), WORLD_H
                elif edge == "left":
                    p.x, p.y = 0.0, rng.uniform(1.0, WORLD_H - 1.0)
                else:
                    p.x, p.y = WORLD_W, rng.uniform(1.0, WORLD_H - 1.0)
                p.active     = True
                p.driven_off = False
                p.attack_cd  = 0.0
                p.target_id  = None
                p.kills      = 0
                p.spawn_time = world.t  # track when predator appeared
        return

    # Grace period: predator can't be driven off in the first 5 seconds
    grace = getattr(p, "spawn_time", 0.0)
    time_active = world.t - grace

    # Target selection: distance weighted by stealth (high stealth = harder to detect)
    best, best_score = None, float("inf")
    for c in world.creatures.values():
        d = distance(p.x, p.y, c.x, c.y)
        stealth = c.bear_strength("hide conceal stealth avoid detection", ["stealth", "predator"])
        effective_d = d * (1.0 + stealth * 2.0)
        if effective_d < best_score:
            best_score = effective_d
            best = c

    if best is None:
        p.active = False
        p.spawn_at = world.t + world.sim_params.get("predator_interval", PREDATOR_SPAWN_INTERVAL)
        return

    p.target_id = best.id

    # Drive-off: requires creatures actively rallying (BEAR-driven intent),
    # not just standing nearby. Grace period prevents instant dismissal.
    if time_active >= 5.0:
        rallied = [
            c for c in world.creatures.values()
            if distance(p.x, p.y, c.x, c.y) <= PREDATOR_DRIVE_OFF_RANGE
            and c.intent == "rallying"
            and not c.fading
        ]
        if len(rallied) >= PREDATOR_DRIVE_OFF_COUNT:
            p.active     = False
            p.driven_off = True
            p.spawn_at   = world.t + PREDATOR_SPAWN_INTERVAL
            for r in rallied:
                r.effect    = "shield"
                r.happiness = min(100.0, r.happiness + 12.0)
            return

    dx, dy = best.x - p.x, best.y - p.y
    dist = math.hypot(dx, dy)
    if dist > 0.05:
        step = PREDATOR_SPEED * world.amplifiers["predator"] * dt
        if step >= dist:
            p.x, p.y = best.x, best.y
        else:
            ratio = step / dist
            p.x += dx * ratio
            p.y += dy * ratio
        p.heading = math.atan2(dy, dx)

    p.attack_cd = max(0.0, p.attack_cd - dt)
    if dist <= PREDATOR_ATTACK_RANGE and p.attack_cd <= 0:
        pred_dmg = world.sim_params.get("predator_damage", PREDATOR_ATTACK_DAMAGE)
        best.vitality  = max(0.0, best.vitality - pred_dmg * world.amplifiers["predator"])
        best.happiness = max(0.0, best.happiness - 20.0)
        best.effect    = "sweat"
        p.attack_cd    = PREDATOR_ATTACK_COOLDOWN
        # Track kills — predator leaves after eating enough
        if best.vitality <= 0:
            p.kills += 1
            if p.kills >= p.max_kills:
                p.active = False
                p.spawn_at = world.t + world.sim_params.get("predator_interval", PREDATOR_SPAWN_INTERVAL)
                p.kills = 0


def _tick_items(world: World, dt: float) -> None:
    for item in world.items.values():
        if item.type == "food" and not item.active:
            if world.t >= item.respawn_at:
                item.active = True
            continue

        if item.type == "ball":
            item.x += item.vx * dt
            item.y += item.vy * dt
            item.x, item.y = clamp_pos(item.x, item.y)
            if item.x <= 0.5 or item.x >= WORLD_W - 0.5:
                item.vx *= -0.8
            if item.y <= 0.5 or item.y >= WORLD_H - 0.5:
                item.vy *= -0.8
            decay = BALL_FRICTION ** dt
            item.vx *= decay
            item.vy *= decay
            if math.hypot(item.vx, item.vy) < 0.05:
                item.vx = item.vy = 0.0

    for c in world.creatures.values():
        for item in world.items.values():
            if not item.active:
                continue
            d = distance(c.x, c.y, item.x, item.y)

            if item.type == "food" and d < FOOD_EAT_DISTANCE:
                if c.happiness < FOOD_EAT_MIN_HAPPY:
                    c.happiness = min(100.0, c.happiness + FOOD_HAPPINESS_BOOST)
                    c.energy    = min(ENERGY_MAX, c.energy + FOOD_ENERGY)
                    c.effect = "sparkles"
                    item.active = False
                    item.respawn_at = world.t + FOOD_RESPAWN_TIME

            elif item.type == "flower" and d < FLOWER_RANGE:
                c.happiness = min(100.0, c.happiness + FLOWER_HAPPY_RATE * dt)

            elif item.type == "ball" and d < BALL_PUSH_DIST:
                speed = math.hypot(c.vx, c.vy)
                if speed > 0.1:
                    nx, ny = c.vx / speed, c.vy / speed
                    item.vx += nx * BALL_PUSH_STRENGTH
                    item.vy += ny * BALL_PUSH_STRENGTH
                    c.happiness = min(100.0, c.happiness + 0.5)


_SOCIAL_RANGE = 4.5


def _tick_social(world: World) -> None:
    for speaker in world.creatures.values():
        if not speaker.speak_to:
            continue
        listener = world.creatures.get(speaker.speak_to)
        if listener is None:
            speaker.speak_to = None
            continue
        d = distance(speaker.x, speaker.y, listener.x, listener.y)
        if d > _SOCIAL_RANGE:
            speaker.speak_to = None
            continue
        if not listener.target:
            dx = speaker.x - listener.x
            dy = speaker.y - listener.y
            if math.hypot(dx, dy) > 0.15:
                listener.heading = math.atan2(dy, dx)


def _auto_regulate_population(world: World) -> None:
    """Adjust breeding and predator amplifiers to steer population toward target."""
    if not world.auto_regulate:
        return
    pop = len(world.creatures)
    target = world.auto_regulate_target
    if target <= 0:
        return

    # ratio > 1 means overpopulated, < 1 means underpopulated
    ratio = pop / target

    # Breeding amplifier: scale down when overpopulated, up when under
    # Clamp to [0.1, 3.0] to avoid extremes
    breed_amp = max(0.1, min(3.0, 1.0 / ratio))
    world.amplifiers["breeding"] = round(breed_amp, 2)

    # Predator amplifier: increase when overpopulated, decrease when under
    pred_amp = max(0.2, min(3.0, ratio))
    world.amplifiers["predator"] = round(pred_amp, 2)

    # Food amplifier: boost when underpopulated to help recovery
    food_amp = max(0.5, min(2.0, 1.0 / (ratio ** 0.5)))
    world.amplifiers["food"] = round(food_amp, 2)


def _check_breeding(world: World, rng: random.Random) -> None:
    if world.breed_queue is None:
        return
    pop = len(world.creatures)
    if pop >= MAX_POPULATION:
        return

    # Soft cap: breeding probability tapers near the cap but never reaches zero
    # until the hard cap. This ensures continuous generational overlap.
    soft_threshold = MAX_POPULATION * 0.6
    if pop > soft_threshold:
        pop_factor = max(0.15, 1.0 - (pop - soft_threshold) / (MAX_POPULATION - soft_threshold))
    else:
        pop_factor = 1.0

    predator_active = world.predator is not None and world.predator.active
    if predator_active:
        return

    creatures = list(world.creatures.values())
    triggered: set[str] = set()

    for i, a in enumerate(creatures):
        if a.id in triggered:
            continue
        breed_happiness = world.sim_params["breed_happiness"]
        breed_dist = world.sim_params["breed_distance"]
        breed_cd = world.sim_params["breed_cooldown"]
        if a.is_rabid or not a.can_breed(breed_happiness):
            continue
        for b in creatures[i + 1:]:
            if b.id in triggered:
                continue
            if b.is_rabid or not b.can_breed(breed_happiness):
                continue
            d = distance(a.x, a.y, b.x, b.y)
            if d <= breed_dist:
                # Breed probability based on behavior profile
                # Use mating gene text directly — no required tag gate
                a_breed = a.bear_strength("mate reproduce offspring eager breed", [])
                b_breed = b.bear_strength("mate reproduce offspring eager breed", [])
                avg_drive = (a_breed + b_breed) / 2
                breed_prob = avg_drive * world.amplifiers["breeding"] * pop_factor

                if rng.random() < breed_prob:
                    triggered.add(a.id)
                    triggered.add(b.id)
                    a.breed_cooldown = breed_cd
                    b.breed_cooldown = breed_cd
                    a.energy -= 15.0
                    b.energy -= 15.0
                    a.children_count += 1
                    b.children_count += 1
                    a.effect = "hearts"
                    b.effect = "hearts"
                    world.breed_queue.put_nowait(("breed", a.id, b.id))
                    break


# ---------------------------------------------------------------------------
# Fight resolution (when creatures are in interaction range)
# ---------------------------------------------------------------------------

FIGHT_RANGE = 1.5


def check_fights(world: World, rng: random.Random) -> None:
    """Check for fights between nearby creatures based on combat behavior.

    Fights only occur when a creature is rabid (enraged) or has very high
    combat drive. Normal proximity does not trigger fights — creatures
    are social by default, not hostile.
    """
    creatures = list(world.creatures.values())
    for i, a in enumerate(creatures):
        if not a.is_rabid and a.bear_strength("fight aggression territorial combat", ["combat", "nearby"]) < 0.45:
            continue  # only rabid or very aggressive creatures initiate
        for b in creatures[i + 1:]:
            d = distance(a.x, a.y, b.x, b.y)
            if d > FIGHT_RANGE:
                continue

            a_combat = a.bear_strength("fight aggression territorial combat", ["combat", "nearby"])
            a_combat += world.epoch.aggression_bonus

            # Much lower fight probability — fights are rare events
            fight_chance = a_combat * 0.05 if a.is_rabid else a_combat * 0.01
            if rng.random() < fight_chance and a.stats:
                defense = b.stats.defense if b.stats else 0.3
                dmg = BASE_DAMAGE * a.stats.damage * (1 - defense * DEFENSE_FACTOR) * 0.5
                dmg = max(1.0, dmg)
                b.hp -= dmg
                b.effect = "sweat"
                if b.hp <= 0:
                    a.kills += 1
