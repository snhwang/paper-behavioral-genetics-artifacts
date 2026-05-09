"""BEAR Brain Engine for Evolutionary Ecosystem — Dual-Path Decision System.

Slow path: LLM generates dialogue and complex interactions on trigger events.
Fast path lives in sim.py (behavior profile scores drive movement every tick).

Architecture:
- CreatureAgent: implements EntityAgent ABC for a single Creature
- BrainEngine: async loop, fires concurrent brain ticks across all agents
- Markers: [!wander], [!approach(id=X)], [!roll], [!circle], [!sprint],
           [!sneak], [!effect(X)], [!thought(text)], [!breed(id=X)]
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from bear import Corpus, Retriever, Context
from bear.config import Config
from bear.models import Instruction, InstructionType, ScopeCondition, ScoredInstruction
from bear.llm import LLM
from bear.retriever import Embedder

from .sim import (
    Creature, World,
    SPEED_MAP, VALID_ANIMATIONS, VALID_MOODS, VALID_EFFECTS,
    WORLD_W, WORLD_H, distance,
    PREDATOR_DRIVE_OFF_RANGE, BREED_DISTANCE, BREED_COOLDOWN, BREED_HAPPINESS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Item BEAR corpus
# ---------------------------------------------------------------------------

_ITEM_INSTRUCTIONS = [
    ("ball",    ["ball_nearby", "play"],
     "The ball is here! Kick it, chase it, and invite others to play — it's irresistible. "
     "You want to kick the ball or suggest playing with it."),
    ("food",    ["food_nearby", "eat", "hungry"],
     "There is food nearby! You feel hungry and want to eat it right away. "
     "Eating will make you feel much better."),
    ("flower",  ["flower_nearby", "beauty", "calm"],
     "A pretty flower is here. It makes you feel calm and peaceful. "
     "You could stop to admire it or mention how nice it smells."),
    ("tree",    ["tree_nearby", "shelter", "rest"],
     "A large tree stands nearby — good for shade, resting under, or meeting by. "
     "You could suggest resting here or using it as a landmark."),
    ("rock",    ["rock_nearby", "explore"],
     "A rock is nearby — solid and unmovable. You could sit on it, hide behind it, "
     "or just notice it as part of the landscape."),
]

_ITEM_CORPUS: Corpus | None = None
_ITEM_RETRIEVER: Retriever | None = None


def _get_item_retriever(config: Config, embedder: Embedder | None) -> Retriever:
    global _ITEM_CORPUS, _ITEM_RETRIEVER
    if _ITEM_RETRIEVER is not None:
        return _ITEM_RETRIEVER

    _ITEM_CORPUS = Corpus()
    for item_type, scope_tags, content in _ITEM_INSTRUCTIONS:
        _ITEM_CORPUS.add(Instruction(
            id       = f"item_{item_type}",
            type     = InstructionType.DIRECTIVE,
            priority = 50,
            content  = content,
            scope    = ScopeCondition(tags=scope_tags),
            tags     = [item_type] + scope_tags,
        ))

    _ITEM_RETRIEVER = Retriever(_ITEM_CORPUS, config=config)
    if embedder is not None:
        _ITEM_RETRIEVER._embedder = embedder
    _ITEM_RETRIEVER.build_index()
    return _ITEM_RETRIEVER


# ---------------------------------------------------------------------------
# Marker parsing
# ---------------------------------------------------------------------------

MARKER_RE = re.compile(r"\[!(\w+)(?:\(([^)]*)\))?\]")


def parse_markers(text: str) -> list[tuple[str, str | None]]:
    return [(m.group(1), m.group(2)) for m in MARKER_RE.finditer(text)]


def strip_markers(text: str) -> str:
    cleaned = MARKER_RE.sub("", text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def markers_to_decision(
    markers: list[tuple[str, str | None]],
    creature: Creature,
    world: World,
    rng: random.Random,
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for command, args_str in markers:
        args = args_str.strip() if args_str else ""

        if command == "wander":
            result["target_x"] = rng.uniform(1.0, WORLD_W - 1.0)
            result["target_y"] = rng.uniform(1.0, WORLD_H - 1.0)
            result["animation"] = "walk"
            result["intent"] = "wandering"

        elif command == "approach":
            target_id = None
            if args.startswith("id="):
                target_id = args[3:].strip()
            elif args == "nearest":
                best, best_d = None, float("inf")
                for other in world.creatures.values():
                    if other.id == creature.id:
                        continue
                    d = distance(creature.x, creature.y, other.x, other.y)
                    if d < best_d:
                        best_d = d
                        best = other
                if best:
                    target_id = best.id
            elif args.startswith("item="):
                item_type = args[5:].strip().lower()
                best_x, best_y, best_d = None, None, float("inf")
                # Search static world items
                for item in world.items.values():
                    if item.type == item_type and item.active:
                        d = distance(creature.x, creature.y, item.x, item.y)
                        if d < best_d:
                            best_d = d
                            best_x, best_y = item.x, item.y
                # Also search spawned food for "food" type
                if item_type == "food":
                    for food in world.food:
                        d = distance(creature.x, creature.y, food.x, food.y)
                        if d < best_d:
                            best_d = d
                            best_x, best_y = food.x, food.y
                if best_x is not None:
                    result["target_x"] = best_x
                    result["target_y"] = best_y
                    result["animation"] = "walk"
                    result["intent"]    = f"going to {item_type}"
                    dx = best_x - creature.x
                    dy = best_y - creature.y
                    if math.hypot(dx, dy) > 0.1:
                        result["face_heading"] = math.atan2(dy, dx)

            if target_id and target_id in world.creatures:
                other = world.creatures[target_id]
                result["target_x"] = other.x
                result["target_y"] = other.y
                result["animation"] = "walk"
                result["intent"] = f"approaching {other.name}"
                dx = other.x - creature.x
                dy = other.y - creature.y
                if math.hypot(dx, dy) > 0.1:
                    result["face_heading"] = math.atan2(dy, dx)

        elif command == "roll":
            result["animation"] = "roll"
            result["intent"] = "rolling"

        elif command == "circle":
            cx, cy = creature.x, creature.y
            radius = 1.5
            points = []
            for i in range(8):
                angle = 2 * math.pi * i / 8
                px = max(0.5, min(WORLD_W - 0.5, cx + radius * math.cos(angle)))
                py = max(0.5, min(WORLD_H - 0.5, cy + radius * math.sin(angle)))
                points.append({"x": px, "y": py})
            result["circle_waypoints"] = points
            result["animation"] = "circle"
            result["intent"] = "circling"

        elif command == "sprint":
            result["animation"] = "sprint"
            result["intent"] = "sprinting"
            if not result.get("target_x"):
                result["target_x"] = rng.uniform(1.0, WORLD_W - 1.0)
                result["target_y"] = rng.uniform(1.0, WORLD_H - 1.0)

        elif command == "sneak":
            result["animation"] = "sneak"
            result["intent"] = "sneaking"
            if not result.get("target_x"):
                result["target_x"] = rng.uniform(1.0, WORLD_W - 1.0)
                result["target_y"] = rng.uniform(1.0, WORLD_H - 1.0)

        elif command == "effect":
            if args in VALID_EFFECTS:
                result["effect"] = args

        elif command == "thought":
            if args and len(args) <= 80:
                result["thought"] = args

        elif command == "busy":
            try:
                val = float(args)
                result["busy_seconds"] = max(0.0, min(10.0, val))
            except (ValueError, TypeError):
                pass

        elif command == "happiness":
            try:
                val = int(float(args))
                result["happiness_delta"] = max(-20, min(20, val))
            except (ValueError, TypeError):
                pass

        elif command == "mood":
            if args in VALID_MOODS:
                result["mood"] = args

        elif command == "breed":
            if args == "nearest" or not args:
                result["breed_nearest"] = True
            elif args.startswith("id="):
                result["breed_target"] = args[3:].strip()

        elif command == "challenge":
            if args == "nearest" or not args:
                result["challenge_nearest"] = True
            elif args.startswith("id="):
                result["challenge_target"] = args[3:].strip()

        elif command == "flee":
            result["animation"] = "sprint"
            result["intent"]    = "fleeing"
            result["flee"]      = True

        elif command == "rally":
            result["animation"] = "sprint"
            result["intent"]    = "rallying"
            result["rally"]     = True

        else:
            logger.debug("Unknown marker: %s", command)

    return result


# ---------------------------------------------------------------------------
# Decision dataclass
# ---------------------------------------------------------------------------


@dataclass
class CreatureDecision:
    target_x:         float | None       = None
    target_y:         float | None       = None
    animation:        str | None         = None
    mood:             str | None         = None
    effect:           str | None         = None
    thought:          str | None         = None
    speak_to:         str | None         = None
    face_heading:     float | None       = None
    intent:           str                = ""
    busy_seconds:     float              = 0.0
    happiness_delta:  int                = 0
    circle_waypoints: list[dict] | None  = None
    breed_target:     str | None         = None
    breed_nearest:    bool               = False
    challenge_target: str | None         = None
    challenge_nearest: bool              = False
    flee:             bool               = False
    rally:            bool               = False


# ---------------------------------------------------------------------------
# EntityAgent ABC
# ---------------------------------------------------------------------------


class EntityAgent(ABC):
    def __init__(self, entity_id: str):
        self.entity_id           = entity_id
        self.last_trigger:   str | None = None
        self.last_decision_time: float  = 0.0

    @abstractmethod
    def get_creature(self, world: World) -> Creature | None: ...

    @abstractmethod
    def is_busy(self, world: World) -> bool: ...

    @abstractmethod
    def determine_trigger(self, world: World) -> str: ...

    @abstractmethod
    def needs_update(self, world: World, trigger: str) -> bool: ...

    @abstractmethod
    def build_context(self, world: World, trigger: str) -> Context: ...

    @abstractmethod
    def build_query(self, world: World, trigger: str) -> str: ...

    @abstractmethod
    def get_system_prompt(self, guidance: str) -> str: ...

    @abstractmethod
    def make_decision(self, text: str, world: World) -> CreatureDecision: ...

    @abstractmethod
    def make_fallback_decision(
        self, scored: list[ScoredInstruction], world: World,
    ) -> CreatureDecision: ...

    @abstractmethod
    def apply_decision(
        self, world: World, decision: CreatureDecision, current_time: float,
    ) -> None: ...


# ---------------------------------------------------------------------------
# NPC system prompt — includes epoch/weather/energy/HP context
# ---------------------------------------------------------------------------

NPC_SYSTEM_PROMPT = """\
You are {name}. Personality: {personality}. Social style: {social_style}.
Mood: {mood} | Happiness: {happiness:.0f}/100 | Energy: {energy:.0f}/100 | HP: {hp:.0f}/100
Environment: Epoch={epoch}, Weather={weather}
STANDING RULE: Always watch for predator warnings. If a predator is attacking, \
IMMEDIATELY override all other behavior and use [!flee] or [!rally] — no exceptions.
{items_context}{priority_alert}=== BEHAVIORAL DIRECTIVE ===
{guidance}
===
{social_context}
OUTPUT FORMAT — follow exactly:
1. Write the words {name} speaks aloud (dialogue only, 8 words or fewer).
   - No narration, no stage directions, no quotes, no name prefix like "{name} said:"
   - You MAY mention items from "Nearby" above; do NOT invent anything else
   - Personality shapes tone (bold=blunt, shy=hesitant, hostile=cold, friendly=warm)
2. On the SAME line, add exactly 1-2 closed markers:
   Normal: [!approach(nearest)] [!approach(item=ball)] [!approach(item=food)] [!approach(item=flower)]
   [!approach(item=tree)] [!wander] [!roll] [!circle] [!sneak] [!sprint]
   [!mood(happy|playful|cautious|excited|annoyed)] [!effect(hearts|sparkles|exclamation|question_mark)]
   Danger: [!flee] [!rally] [!challenge(nearest)]

CORRECT examples (copy this format exactly):
  Hey Zara, good to see you! [!approach(nearest)] [!mood(happy)]
  Leave me alone. [!sneak] [!mood(annoyed)]
  That food looks tasty! [!approach(item=food)] [!mood(playful)]
  DANGER! Run everyone! [!flee] [!mood(cautious)]
  Stand together, drive it off! [!rally] [!mood(excited)]
  Back off! This is mine! [!challenge(nearest)] [!mood(annoyed)]
WRONG (do not do this):
  "Hey Zara!" — no quotes
  Zara says happily, "..." — no narration
  The butterflies flutter... — no invented things
"""

NPC_GENERIC_PROMPT = """\
You are {name}. Mood: {mood} | Happiness: {happiness:.0f}/100
{items_context}
{social_context}
OUTPUT FORMAT — follow exactly:
1. Write the words {name} speaks aloud (dialogue only, 8 words or fewer).
   - No narration, no stage directions, no quotes
   - You MAY mention items from "Nearby"; do NOT invent anything else
2. On the SAME line, add exactly 1-2 closed markers:
   [!approach(nearest)] [!approach(item=ball)] [!approach(item=food)] [!approach(item=flower)]
   [!approach(item=tree)] [!wander] [!mood(happy|playful|cautious|excited|annoyed)]

CORRECT examples:
  Hi there, nice to see you! [!approach(nearest)] [!mood(happy)]
  I feel like wandering. [!wander] [!mood(neutral)]
"""

NPC_SOCIAL_GREETING = """\
{other_name} just walked up to you. Greet them. ({other_name}: {other_personality})
"""

NPC_SOCIAL_CHAT = """\
{other_name} said: "{other_speech}" — reply to them. ({other_name}: {other_personality})
"""

NPC_SOCIAL_HAPPY_PAIR = """\
You and {other_name} are close and both happy. Show warmth toward {other_name}.
"""

NPC_SOCIAL_CHAT_IDLE = """\
{other_name} is hanging out nearby. Make casual conversation — comment on the \
weather, your surroundings, how you're feeling, or just say something friendly. \
({other_name}: {other_personality})
"""


# ---------------------------------------------------------------------------
# CreatureAgent
# ---------------------------------------------------------------------------

IDLE_REEVAL_INTERVAL   = 2.0   # BEAR retrieval drives movement, needs to be responsive
SOCIAL_REEVAL_INTERVAL = 12.0
SOCIAL_RANGE           = 4.0


class CreatureAgent(EntityAgent):
    """BEAR-driven behavioral agent for one NPC creature."""

    def __init__(
        self,
        creature_id: str,
        retriever: Retriever,
        llm: LLM | None,
        rng: random.Random,
    ):
        super().__init__(creature_id)
        self.retriever = retriever
        self.llm       = llm
        self.rng       = rng
        self._last_creature: Creature | None = None
        self._greeted: dict[str, float] = {}
        self._social_target: str | None = None

    def get_creature(self, world: World) -> Creature | None:
        return world.creatures.get(self.entity_id)

    def is_busy(self, world: World) -> bool:
        c = self.get_creature(world)
        return c.is_busy(world.t) if c else False

    def _nearest_other(self, world: World) -> tuple[Creature | None, float]:
        c = self.get_creature(world)
        if not c:
            return None, float("inf")
        best, best_d = None, float("inf")
        for other in world.creatures.values():
            if other.id == c.id:
                continue
            d = distance(c.x, c.y, other.x, other.y)
            if d < best_d:
                best_d = d
                best = other
        return best, best_d

    def determine_trigger(self, world: World) -> str:
        c = self.get_creature(world)
        if not c:
            return "idle"

        if c.is_rabid:
            return "enraged"

        if world.predator and world.predator.active:
            pd = distance(c.x, c.y, world.predator.x, world.predator.y)
            if pd < 8.0:
                return "predator"

        other, d = self._nearest_other(world)
        if other is None:
            return "idle"

        # Breed trigger takes priority when both creatures are ready and close
        breed_dist = world.sim_params.get("breed_distance", BREED_DISTANCE)
        breed_happy = world.sim_params.get("breed_happiness", BREED_HAPPINESS)
        if (d < breed_dist and c.can_breed(breed_happy)
                and other.can_breed(breed_happy) and not other.is_rabid):
            return "breed"

        if d < SOCIAL_RANGE:
            self._social_target = other.id
            last_t = self._greeted.get(other.id, 0.0)
            if world.t - last_t >= SOCIAL_REEVAL_INTERVAL:
                return "greeting"
            if c.happiness >= 35 and world.t - self.last_decision_time >= 15.0:
                return "chat"

        return "idle"

    def needs_update(self, world: World, trigger: str) -> bool:
        if trigger != self.last_trigger:
            return True
        if trigger in ("greeting", "chat"):
            return True
        if trigger == "breed":
            return world.t - self.last_decision_time >= 5.0
        if trigger == "enraged":
            return world.t - self.last_decision_time >= 3.0
        if trigger == "predator":
            return world.t - self.last_decision_time >= 1.0
        if trigger == "other_nearby":
            return world.t - self.last_decision_time >= 6.0
        if trigger == "idle":
            return world.t - self.last_decision_time >= IDLE_REEVAL_INTERVAL
        return False

    def build_context(self, world: World, trigger: str) -> Context:
        c = self.get_creature(world)
        if not c:
            return Context(tags=["idle"], domain="evolutionary_ecosystem")

        tags = [trigger]
        tags.append(f"mood_{c.mood}")

        # Always include general behavioral context
        tags += ["wander", "general"]

        if c.happiness >= 75:
            tags.append("happy")
        elif c.happiness < 35:
            tags.append("unhappy")

        if c.skills.can_roll:     tags.append("skill_rolling")
        if c.skills.tight_circle: tags.append("skill_circle")
        if c.skills.bounce_gait:  tags.append("skill_bounce")
        if c.skills.can_sneak:    tags.append("skill_sneak")

        if trigger in ("greeting", "other_nearby", "happy_pair", "chat", "breed"):
            tags.append("social")

        if trigger == "enraged":
            tags += ["enraged", "aggression", "nearby"]

        if trigger == "predator":
            tags += ["predator", "danger"]
            if world.predator and world.predator.active:
                pd = distance(c.x, c.y, world.predator.x, world.predator.y)
                if pd < PREDATOR_DRIVE_OFF_RANGE:
                    tags.append("rally")
                else:
                    tags.append("flee")

        # Epoch and weather as context tags
        tags.append(f"epoch:{world.epoch.name}")
        tags.append(f"weather:{world.weather}")

        # Energy and health state — included as context for BEAR to weigh
        if c.energy < 70:
            tags.append("food")
            tags.append("foraging")
        if c.energy < 40:
            tags.append("hunger")
            tags.append("survival")
        if c.hp < 50:
            tags.append("danger")
            tags.append("survival")

        # Breeding readiness — context for BEAR to retrieve breeding genes
        breed_dist = world.sim_params.get("breed_distance", BREED_DISTANCE)
        breed_happy = world.sim_params.get("breed_happiness", BREED_HAPPINESS)
        if c.can_breed(breed_happy):
            tags.append("breed")
            other, d = self._nearest_other(world)
            if other and d < breed_dist and other.can_breed(breed_happy):
                tags.append("happy_pair")

        # Nearby items (static + spawned food)
        for item in world.items.values():
            if item.active and distance(c.x, c.y, item.x, item.y) < 3.5:
                tags.append(f"{item.type}_nearby")
        if any(distance(c.x, c.y, f.x, f.y) < 3.5 for f in world.food):
            tags.append("food_nearby")

        return Context(tags=tags, domain="evolutionary_ecosystem")

    def build_query(self, world: World, trigger: str) -> str:
        c = self.get_creature(world)
        if not c:
            return "What should I do?"

        other, d = self._nearest_other(world)

        # Include epoch/weather context in queries
        env_ctx = f" [Epoch: {world.epoch.name}, Weather: {world.weather}]"

        if trigger == "greeting" and other:
            other_said = f' They just said: "{other.last_speech}"' if other.last_speech else ""
            return (f"I'm meeting {other.name} for the first time (distance {d:.1f}).{other_said} "
                    f"How do I react given my personality?{env_ctx}")
        elif trigger == "happy_pair" and other:
            return (f"I'm very happy and {other.name} is nearby and happy too. "
                    f"How do I express this joyful connection?{env_ctx}")
        elif trigger == "chat" and other:
            other_said = f' {other.name} just said: "{other.last_speech}"' if other.last_speech else ""
            return (f"{other.name} is hanging out nearby.{other_said} "
                    f"Make casual conversation.{env_ctx}")
        elif trigger == "other_nearby" and other:
            other_said = f' {other.name} said: "{other.last_speech}"' if other.last_speech else ""
            return f"{other.name} is nearby (dist {d:.1f}).{other_said} What do I do?{env_ctx}"
        elif trigger == "enraged":
            return (f"I am ENRAGED (rage={c.rage:.0f}/100). "
                    f"I must challenge the nearest creature. What do I do?{env_ctx}")
        elif trigger == "breed" and other:
            return (f"I feel a warm social bond with {other.name} nearby (distance {d:.1f}). "
                    f"We are happy together. How do I express this connection?{env_ctx}")
        elif trigger == "predator":
            pd_str = ""
            if world.predator and world.predator.active:
                pd = distance(c.x, c.y, world.predator.x, world.predator.y)
                pd_str = f" (predator at dist {pd:.1f})"
            return f"A predator is attacking!{pd_str} Do I flee or rally others to defend?{env_ctx}"
        else:
            nearby_items = [
                item.type for item in world.items.values()
                if item.active and distance(c.x, c.y, item.x, item.y) < 3.5
            ]
            if any(distance(c.x, c.y, f.x, f.y) < 3.5 for f in world.food):
                nearby_items.append("food")
            item_str = f" I can see: {', '.join(nearby_items[:3])}." if nearby_items else ""
            energy_str = f" Energy: {c.energy:.0f}/100." if c.energy < 70 else ""
            return (f"I'm feeling {c.mood} with happiness {c.happiness:.0f}/100.{energy_str}"
                    f"{item_str} What should I do?{env_ctx}")

    def _get_generic_prompt(self) -> str:
        c = self._last_creature
        if not c:
            return "Act naturally."
        social_ctx = ""
        trigger = self.last_trigger or "idle"
        world_ref = getattr(self, "_world_ref", None)
        if self._social_target and world_ref:
            other = world_ref.creatures.get(self._social_target)
            if other:
                other_speech = other.last_speech or ""
                if trigger == "greeting" or not other_speech:
                    social_ctx = NPC_SOCIAL_GREETING.format(
                        other_name=other.name,
                        other_personality="unknown",
                    )
                else:
                    social_ctx = NPC_SOCIAL_CHAT.format(
                        other_name=other.name,
                        other_speech=other_speech,
                        other_personality="unknown",
                    )
        items_ctx = ""
        if world_ref and hasattr(world_ref, "items"):
            nearby = [
                item.type for item in world_ref.items.values()
                if item.active and distance(c.x, c.y, item.x, item.y) < 3.5
            ]
            if nearby:
                items_ctx = f"Nearby: {', '.join(nearby[:5])}."
        return NPC_GENERIC_PROMPT.format(
            name           = c.name,
            mood           = c.mood,
            happiness      = c.happiness,
            items_context  = items_ctx,
            social_context = social_ctx,
        )

    def get_system_prompt(self, guidance: str, trigger: str = "") -> str:
        c = self._last_creature
        if not c:
            return guidance

        world_ref = getattr(self, "_world_ref", None)

        # Build social context if applicable
        social_ctx = ""
        trigger = self.last_trigger or "idle"
        if self._social_target and world_ref:
            other = world_ref.creatures.get(self._social_target)
            if other:
                other_speech = other.last_speech or ""
                other_personality = other.genes.get("personality", "unknown")[:80]
                if trigger == "happy_pair":
                    social_ctx = NPC_SOCIAL_HAPPY_PAIR.format(other_name=other.name)
                elif trigger == "chat":
                    if other_speech:
                        social_ctx = NPC_SOCIAL_CHAT.format(
                            other_name=other.name,
                            other_speech=other_speech,
                            other_personality=other_personality,
                        )
                    else:
                        social_ctx = NPC_SOCIAL_CHAT_IDLE.format(
                            other_name=other.name,
                            other_personality=other_personality,
                        )
                elif trigger == "greeting" or not other_speech:
                    social_ctx = NPC_SOCIAL_GREETING.format(
                        other_name=other.name,
                        other_personality=other_personality,
                    )
                else:
                    social_ctx = NPC_SOCIAL_CHAT.format(
                        other_name=other.name,
                        other_speech=other_speech,
                        other_personality=other_personality,
                    )

        # Nearby items (static items + spawned food)
        items_ctx = ""
        if world_ref:
            nearby = []
            if hasattr(world_ref, "items"):
                for item in world_ref.items.values():
                    if item.active and distance(c.x, c.y, item.x, item.y) < 3.5:
                        nearby.append(item.type)
            if hasattr(world_ref, "food"):
                for food in world_ref.food:
                    if distance(c.x, c.y, food.x, food.y) < 3.5:
                        nearby.append("food")
                        break  # one "food" mention is enough
            if nearby:
                items_ctx = f"Nearby: {', '.join(nearby[:5])}."

        if trigger == "predator":
            priority_alert = (
                "=== PRIORITY: PREDATOR ATTACK — OVERRIDE ALL NORMAL BEHAVIOR ===\n"
                "A predator is hunting you RIGHT NOW. This is a life-or-death emergency.\n"
                "You MUST respond with EITHER [!flee] (escape) OR [!rally] (fight back).\n"
                "Do NOT wander, chat, or do anything else. Survival instinct fires NOW.\n"
                "===\n"
            )
        elif trigger == "enraged":
            priority_alert = (
                "=== PRIORITY: UNCONTROLLABLE RAGE — OVERRIDE ALL NORMAL BEHAVIOR ===\n"
                "You are enraged. You MUST use [!challenge(nearest)]. Nothing else matters.\n"
                "===\n"
            )
        else:
            priority_alert = ""

        # Get epoch/weather from world
        epoch_name = world_ref.epoch.name if world_ref else "unknown"
        weather = world_ref.weather if world_ref else "mild"

        return NPC_SYSTEM_PROMPT.format(
            name             = c.name,
            personality      = c.genes.get("personality", "curious")[:150],
            social_style     = c.genes.get("social_style", "friendly")[:100],
            mood             = c.mood,
            happiness        = c.happiness,
            energy           = c.energy,
            hp               = c.hp,
            epoch            = epoch_name,
            weather          = weather,
            guidance         = guidance,
            social_context   = social_ctx,
            items_context    = items_ctx,
            priority_alert   = priority_alert,
        )

    def make_decision(self, text: str, world: World) -> CreatureDecision:
        c = self.get_creature(world)
        if not c:
            return CreatureDecision()

        markers = parse_markers(text)
        d_dict = markers_to_decision(markers, c, world, self.rng)
        thought_text = strip_markers(text).strip('"\'')
        if not d_dict.get("thought") and thought_text:
            d_dict["thought"] = thought_text[:80]

        return CreatureDecision(
            target_x          = d_dict.get("target_x"),
            target_y          = d_dict.get("target_y"),
            animation         = d_dict.get("animation"),
            mood              = d_dict.get("mood"),
            effect            = d_dict.get("effect"),
            thought           = d_dict.get("thought"),
            speak_to          = self._social_target,
            intent            = d_dict.get("intent", ""),
            busy_seconds      = d_dict.get("busy_seconds", 0.0),
            happiness_delta   = d_dict.get("happiness_delta", 0),
            circle_waypoints  = d_dict.get("circle_waypoints"),
            breed_target      = d_dict.get("breed_target"),
            breed_nearest     = d_dict.get("breed_nearest", False),
            face_heading      = d_dict.get("face_heading"),
            challenge_target  = d_dict.get("challenge_target"),
            challenge_nearest = d_dict.get("challenge_nearest", False),
            flee              = d_dict.get("flee", False),
            rally             = d_dict.get("rally", False),
        )

    def make_fallback_decision(
        self, scored: list[ScoredInstruction], world: World,
    ) -> CreatureDecision:
        if not scored:
            return CreatureDecision(
                target_x = self.rng.uniform(1.0, WORLD_W - 1.0),
                target_y = self.rng.uniform(1.0, WORLD_H - 1.0),
                animation = "walk",
                intent    = "wandering",
            )
        top_text = scored[0].instruction.content
        return self.make_decision(top_text, world)

    def apply_decision(
        self, world: World, decision: CreatureDecision, current_time: float,
    ) -> None:
        c = self.get_creature(world)
        if not c:
            return

        # Always update intent — clears stale rally/flee from previous ticks
        c.intent = decision.intent or "idle"

        if decision.target_x is not None and decision.target_y is not None:
            c.target = {"x": decision.target_x, "y": decision.target_y}
            c.circle_waypoints = []

        if decision.circle_waypoints:
            c.circle_waypoints = decision.circle_waypoints
            c.circle_idx = 0
            c.target = None

        if decision.animation:
            c.animation = decision.animation

        if decision.mood:
            c.mood = decision.mood

        if decision.effect:
            c.effect = decision.effect

        if decision.thought:
            c.thought = decision.thought
            c.last_speech = decision.thought

        c.speak_to = decision.speak_to

        if decision.face_heading is not None:
            c.heading = decision.face_heading


        if decision.busy_seconds > 0:
            c.busy_until = current_time + decision.busy_seconds

        if decision.happiness_delta != 0:
            c.happiness = max(0.0, min(100.0, c.happiness + decision.happiness_delta))

        # Challenge (rabid attack)
        if decision.challenge_nearest or decision.challenge_target:
            if decision.challenge_nearest:
                best, best_d = self._nearest_other(world)
                target_id = best.id if best else None
            else:
                target_id = decision.challenge_target
            if target_id and target_id in world.creatures:
                world.challenge_queue.append((c.id, target_id))
                c.animation = "sprint"
                c.effect    = "sweat"

        # Flee
        if decision.flee and world.predator and world.predator.active:
            dx = c.x - world.predator.x
            dy = c.y - world.predator.y
            dist = math.hypot(dx, dy)
            if dist > 0.1:
                flee_x = max(1.0, min(WORLD_W - 1.0, c.x + dx / dist * 8.0))
                flee_y = max(1.0, min(WORLD_H - 1.0, c.y + dy / dist * 8.0))
                c.target = {"x": flee_x, "y": flee_y}
                c.animation = "sprint"

        # Rally — group up with nearest ally first, then charge predator together
        if decision.rally and world.predator and world.predator.active:
            nearest, nd = self._nearest_other(world)
            if nearest and nd > 3.0:
                # Too far from allies — approach nearest creature first
                c.target    = {"x": nearest.x, "y": nearest.y}
                c.animation = "sprint"
                c.intent    = f"grouping with {nearest.name}"
            else:
                # Grouped up — charge the predator
                c.target    = {"x": world.predator.x, "y": world.predator.y}
                c.animation = "sprint"

        # Breed — BEAR triggered via [!breed(nearest)] or [!breed(id=X)]
        breed_target_id = decision.breed_target
        if decision.breed_nearest:
            best, best_d = self._nearest_other(world)
            if best:
                breed_target_id = best.id
        if breed_target_id and breed_target_id in world.creatures:
            mate = world.creatures[breed_target_id]
            d = distance(c.x, c.y, mate.x, mate.y)
            breed_dist = world.sim_params.get("breed_distance", BREED_DISTANCE)
            breed_happy = world.sim_params.get("breed_happiness", BREED_HAPPINESS)
            breed_cd = world.sim_params.get("breed_cooldown", BREED_COOLDOWN)
            if d <= breed_dist and c.can_breed(breed_happy) and mate.can_breed(breed_happy) and not mate.is_rabid:
                # Close enough — breed
                if world.breed_queue:
                    c.breed_cooldown = breed_cd
                    mate.breed_cooldown = breed_cd
                    c.energy -= 15.0
                    mate.energy -= 15.0
                    c.children_count += 1
                    mate.children_count += 1
                    c.effect = "hearts"
                    mate.effect = "hearts"
                    world.breed_queue.put_nowait(("breed", c.id, mate.id))
                    logger.info("BREED %s + %s", c.name, mate.name)
            else:
                # Too far — approach the mate
                c.target = {"x": mate.x, "y": mate.y}
                c.animation = "walk"
                c.intent = f"approaching {mate.name}"


# ---------------------------------------------------------------------------
# BrainEngine — async loop for all creatures
# ---------------------------------------------------------------------------


class BrainEngine:
    def __init__(
        self,
        world: World,
        agents: dict[str, CreatureAgent],
        llm: LLM | None,
        rng: random.Random,
        bear_config: Config,
        tick_interval: float = 0.5,
    ):
        self.world         = world
        self.agents        = agents
        self.llm           = llm
        self.rng           = rng
        self.bear_config   = bear_config
        self.tick_interval = tick_interval
        self.bear_disabled = False
        self.stats: dict[str, Any] = {"ticks": 0, "llm_calls": 0, "fallbacks": 0}
        self._thought_callbacks: list = []
        self._action_log: list | None = None  # set externally to enable action logging

    def add_thought_callback(self, cb) -> None:
        self._thought_callbacks.append(cb)

    async def run(self) -> None:
        while True:
            try:
                if not self.world.paused:
                    await self._tick_all()
            except Exception as e:
                logger.error("Brain engine error: %s", e, exc_info=True)
            await asyncio.sleep(self.tick_interval)

    async def _tick_all(self) -> None:
        current_time = self.world.t

        pending: list[tuple[str, CreatureAgent, str]] = []
        for agent_id, agent in self.agents.items():
            c = agent.get_creature(self.world)
            if c is None:
                continue
            # Debug: log why creatures aren't getting updates
            if not pending and self.stats["ticks"] % 20 == 0:
                trigger = agent.determine_trigger(self.world)
                needs = agent.needs_update(self.world, trigger)
                busy = agent.is_busy(self.world)
                logger.debug("BRAIN check %s: trigger=%s needs=%s busy=%s last_t=%.1f now=%.1f",
                            c.name, trigger, needs, busy,
                            agent.last_decision_time, current_time)
            if agent.is_busy(self.world):
                continue
            trigger = agent.determine_trigger(self.world)
            if agent.needs_update(self.world, trigger):
                agent._last_creature = c
                pending.append((agent_id, agent, trigger))

        if not pending:
            if self.stats["ticks"] % 50 == 0 and self.agents:
                sample = next(iter(self.agents.values()))
                sc = sample.get_creature(self.world)
                if sc:
                    t = sample.determine_trigger(self.world)
                    logger.info("BRAIN: %d agents, 0 pending. Sample %s: trigger=%s, last_decision=%.1fs ago",
                                len(self.agents), sc.name, t, current_time - sample.last_decision_time)
            return

        logger.info("BRAIN: %d pending (%s)", len(pending),
                    ", ".join(f"{a.get_creature(self.world).name}:{t}" for _, a, t in pending[:3]
                              if a.get_creature(self.world)))

        tasks = [
            self._brain_tick(agent_id, agent, trigger, current_time)
            for agent_id, agent, trigger in pending
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (agent_id, agent, trigger), result in zip(pending, results):
            if isinstance(result, Exception):
                logger.error("Brain tick error for %s: %s", agent_id, result)
                continue
            decision, trigger_used = result
            c = self.world.creatures.get(agent_id)
            if c:
                tgt = f"({decision.target_x:.1f},{decision.target_y:.1f})" if decision.target_x is not None else "no"
                logger.info("DECISION %s: trigger=%s intent=%s target=%s pos=(%.1f,%.1f) energy=%.0f",
                            c.name, trigger_used, decision.intent, tgt,
                            c.x, c.y, c.energy)

            agent.apply_decision(self.world, decision, current_time)
            agent.last_trigger = trigger_used
            agent.last_decision_time = current_time
            self.stats["ticks"] += 1

            # Log slow-path action tags for analysis
            c2 = self.world.creatures.get(agent_id)
            if c2 and self._action_log is not None:
                entry = {
                    "tick":      self.world.tick_count,
                    "epoch":     self.world.epoch.name,
                    "creature":  c2.name,
                    "trigger":   trigger_used,
                    "flee":      decision.flee,
                    "rally":     decision.rally,
                    "mood":      decision.mood,
                    "animation": decision.animation,
                    "intent":    decision.intent,
                    "breed":     decision.breed_nearest or bool(decision.breed_target),
                }
                self._action_log.append(entry)

            c = self.world.creatures.get(agent_id)
            if c and c.thought:
                for cb in self._thought_callbacks:
                    try:
                        await cb(agent_id, c.thought)
                    except Exception:
                        pass

    async def _brain_tick(
        self,
        agent_id: str,
        agent: CreatureAgent,
        trigger: str,
        current_time: float,
    ) -> tuple[CreatureDecision, str]:
        agent._world_ref = self.world

        ctx   = agent.build_context(self.world, trigger)
        query = agent.build_query(self.world, trigger)

        if trigger == "greeting" and agent._social_target:
            agent._greeted[agent._social_target] = current_time

        bear_off = self.bear_disabled or self.world.bear_disabled

        if bear_off:
            scored  = []
            guidance = "Act naturally given your mood. Interact with others or explore."
            c = self.world.creatures.get(agent_id)
            if c:
                c.last_retrieval = {
                    "text":    "(BEAR disabled — no retrieval)",
                    "score":   0.0,
                    "trigger": trigger,
                    "query":   query[:80],
                }
        else:
            scored = agent.retriever.retrieve(query=query, context=ctx, top_k=5, threshold=0.1)

            c = self.world.creatures.get(agent_id)
            if c and scored:
                top = scored[0]
                gene_cat = (top.instruction.metadata or {}).get("gene_category", "")
                gene_cats = [
                    (s.instruction.metadata or {}).get("gene_category", "")
                    for s in scored
                ]
                c.last_retrieval = {
                    "text":    top.instruction.content,
                    "score":   round(float(top.final_score), 3),
                    "trigger": trigger,
                    "query":   query[:80],
                    "gene":    gene_cat,
                    "locus":   top.instruction.id,
                    "all_genes": gene_cats,
                }

            if not scored:
                agent._social_target = None
                return agent.make_fallback_decision([], self.world), trigger

            # Combine all retrieved genes into guidance — each gene
            # contributes its behavioral influence to the decision
            guidance_parts = []
            seen_cats = set()
            for s in scored:
                cat = (s.instruction.metadata or {}).get("gene_category", "unknown")
                if cat not in seen_cats:
                    seen_cats.add(cat)
                    prompt = (s.instruction.metadata or {}).get("prompt", s.instruction.content)
                    guidance_parts.append(f"[{cat}] {prompt}")
            guidance = "\n".join(guidance_parts)

            item_retriever = _get_item_retriever(self.bear_config, getattr(agent.retriever, "_embedder", None))
            item_scored = item_retriever.retrieve(query=query, context=ctx, top_k=1, threshold=0.15)
            if item_scored:
                item_inst = item_scored[0].instruction.content
                guidance = f"{guidance}\n\nItem directive: {item_inst}"

        # Decide whether to use LLM (for dialogue) or BEAR fallback (for movement).
        # Social triggers need the LLM to generate speech.
        # Movement/idle triggers use BEAR retrieval + marker parsing directly —
        # the gene templates contain embedded action markers like [!wander],
        # [!approach(item=food)], [!flee], etc. that drive movement without LLM.
        llm_triggers = {"greeting", "chat"}
        use_llm = trigger in llm_triggers or bear_off

        if use_llm and self.llm:
            try:
                if bear_off:
                    system = agent._get_generic_prompt()
                else:
                    system = agent.get_system_prompt(guidance, trigger=trigger)
                resp = await self.llm.generate(
                    system=system,
                    user=query,
                    temperature=0.80,
                    max_tokens=60,
                )
                logger.debug("LLM raw [%s] trigger=%s: %s", agent_id, trigger, resp.content)
                decision = agent.make_decision(resp.content, self.world)
                self.stats["llm_calls"] += 1
                agent._social_target = None
                return decision, trigger
            except Exception as e:
                logger.warning("LLM error for %s: %s — using fallback", agent_id, e)

        # BEAR fallback: parse action markers from retrieved gene templates
        self.stats["fallbacks"] += 1
        agent._social_target = None
        if scored:
            # When trigger matches a required_tags instruction, prefer it
            # over the highest-similarity result (which may be a different gene)
            best = scored[0]
            if trigger:
                found_req = False
                for s in scored:
                    req = s.instruction.scope.required_tags
                    if req and trigger in req:
                        best = s
                        found_req = True
                        break
                if not found_req and trigger == "breed":
                    logger.debug("No required_tags=breed instruction in scored results for %s (got %d results, cats: %s)",
                                  agent_id, len(scored),
                                  [((s.instruction.metadata or {}).get("gene_category","?")) for s in scored])
            top_prompt = (best.instruction.metadata or {}).get("prompt", best.instruction.content)
            top_cat = (best.instruction.metadata or {}).get("gene_category", "?")
            logger.debug("BEAR fallback [%s] trigger=%s cat=%s: %s", agent_id, trigger, top_cat, top_prompt[:120])
            return agent.make_decision(top_prompt, self.world), trigger
        logger.debug("BEAR fallback [%s] trigger=%s: NO scored results (tags=%s, query=%s)",
                     agent_id, trigger, ctx.tags, query[:80])
        return agent.make_fallback_decision([], self.world), trigger
