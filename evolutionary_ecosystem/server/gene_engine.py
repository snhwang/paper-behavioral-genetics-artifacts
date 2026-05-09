"""Merged Gene Engine for Evolutionary Ecosystem.

Combines creature ecosystem's NPC gene categories (personality, social_style,
movement_style, reaction_pattern) with spatial evolution's survival categories
(foraging, predator_defense, climate_survival, territorial).

Provides:
- AppearanceParams + extract_appearance() for 3D rendering
- SkillSet + extract_skills() for movement abilities
- EntityStats + extract_stats() for survival stats
- BehaviorProfile + compute_behavior_profile() for BEAR-driven decisions
- Async breeding pipeline with LLM blending/mutation/drift
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

for _ln in ("model2vec", "sentence_transformers", "safetensors", "httpx"):
    logging.getLogger(_ln).setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent))

from bear import (
    Config,
    Corpus,
    EmbeddingBackend,
    Instruction,
    InstructionType,
    ScopeCondition,
)
from bear.evolution import BreedingConfig, breed, express
from bear.llm import LLM
from bear.models import Context, CrossoverMethod, Dominance, GeneLocus, LocusRegistry
from bear.retriever import Embedder, Retriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 8 gene categories (union of creature + spatial)
# ---------------------------------------------------------------------------

GENE_CATEGORIES = [
    # NPC genes (from creature ecosystem)
    "personality",
    "social_style",
    "movement_style",
    "reaction_pattern",
    "mating",
    # Survival genes (from spatial evolution)
    "foraging",
    "predator_defense",
    "climate_survival",
    "territorial",
    "stealth",
    "sensory",
]

_CATEGORY_DESCRIPTIONS = {
    "personality":      "the NPC's general disposition and inner character — curious, shy, bold, impulsive, gentle",
    "social_style":     "how the NPC interacts with others — friendly, aloof, protective, playful, competitive",
    "mating":           "level of interest in reproducing — very eager, eager, moderate, low, or rarely interested",
    "movement_style":   "preferred locomotion and movement quirks — rolling, bouncing, sneaking, sprinting in circles",
    "reaction_pattern": "how the NPC responds to stimuli — investigates, flees, ignores, approaches cautiously",
    "foraging":         "how the lifeform finds, identifies, and acquires food",
    "predator_defense": "how it detects, evades, or fights off predators",
    "climate_survival": "how it copes with harsh weather, floods, drought, and temperature extremes",
    "territorial":      "how it claims, defends, and expands its territory and resources",
    "stealth":          "how well the lifeform conceals itself — camouflage, silent movement, scent masking",
    "sensory":          "acuity of the lifeform's senses — how far and how early it detects predators, food, and other creatures",
}

# Which gene categories feed into the BEAR behavior corpus
BEHAVIOR_CATEGORIES = ["personality", "social_style", "mating", "reaction_pattern",
                       "foraging", "predator_defense", "climate_survival", "territorial",
                       "stealth", "sensory"]

# Priority tiers: survival-critical > environmental > lifestyle
_CATEGORY_PRIORITY = {
    "personality": 40, "social_style": 40, "mating": 70, "reaction_pattern": 40,
    "foraging": 60, "predator_defense": 80, "climate_survival": 60,
    "territorial": 60, "stealth": 80, "sensory": 80,
}
# Which feed into appearance
APPEARANCE_CATEGORIES = ["personality", "social_style", "reaction_pattern", "movement_style"]
MOVEMENT_CATEGORY = "movement_style"

# ---------------------------------------------------------------------------
# Behavior corpus scope tags
# ---------------------------------------------------------------------------

GENE_SCOPE_TAGS: dict[str, list[str]] = {
    "personality":      ["idle", "wander", "mood", "general"],
    "social_style":     ["nearby", "social", "other"],
    "mating":           ["breed", "mate", "reproduce"],
    "reaction_pattern": ["stimulus", "idle", "nearby", "mood"],
    "foraging":         ["food", "hunger", "foraging", "survival"],
    "predator_defense": ["predator", "threat", "defense", "danger"],
    "climate_survival": ["weather", "climate", "survival", "harsh"],
    "territorial":      ["territory", "intruder", "combat", "space"],
    "stealth":          ["stealth", "hide", "camouflage", "predator"],
    "sensory":          ["sense", "detect", "awareness", "predator", "danger"],
}

# ---------------------------------------------------------------------------
# Appearance extraction
# ---------------------------------------------------------------------------

APPEARANCE_PROBES = {
    "body_radius":  "large round plump stocky body with wide girth",
    "head_scale":   "large expressive oversized head with big features",
    "limb_length":  "long slender legs built for running, extended limbs",
    "tail_length":  "long prominent tail, dramatic sweeping tail display",
    "ear_pointy":   "sharp pointed upright ears, alert triangular ears",
}

PERSONALITY_COLOR_PROBES: dict[float, str] = {
    0.0:   "aggressive dominant fierce bold confrontational fighter",
    30.0:  "energetic impulsive excitable adventurous reckless daring",
    60.0:  "friendly warm cheerful outgoing generous welcoming social",
    100.0: "playful curious explorative inventive creative investigative",
    140.0: "gentle calm nurturing patient kind empathetic caring",
    180.0: "thoughtful analytical careful precise observant methodical",
    220.0: "shy reserved timid introverted quiet hesitant cautious",
    260.0: "aloof independent proud self-reliant distant solitary",
    300.0: "moody complex unpredictable brooding mysterious temperamental",
    330.0: "competitive protective territorial possessive assertive jealous",
}

SOCIAL_COLOR_PROBES: dict[float, str] = {
    0.0:   "hostile confrontational unfriendly rude aggressive",
    60.0:  "friendly cooperative sociable welcoming warm",
    120.0: "nurturing protective supportive collaborative",
    180.0: "neutral calm distant reserved professional",
    240.0: "shy withdrawn aloof introverted private",
    300.0: "playful charismatic flirtatious whimsical teasing",
}

REACTION_COLOR_PROBES: dict[float, str] = {
    0.0:   "aggressive bold reactive charges attacks confronts",
    40.0:  "cautious wary defensive tense alert watchful",
    80.0:  "curious investigative explorative approaches examines",
    140.0: "calm gentle careful methodical deliberate slow",
    200.0: "flighty nervous skittish startled retreats flees",
    260.0: "ignores passive neutral unresponsive indifferent",
    300.0: "playful excitable chases reactive bouncy energetic",
    340.0: "social friendly greets embraces welcoming warm",
}

MOVEMENT_COLOR_PROBES: dict[float, str] = {
    0.0:   "fast sprinting charging explosive powerful dash",
    50.0:  "bouncy hopping springy energetic leaping",
    100.0: "rolling tumbling spinning rotating",
    160.0: "steady walking even-paced deliberate plodding",
    220.0: "sneaking creeping stealthy silent tiptoeing",
    280.0: "circling spiraling looping orbital",
    330.0: "floating drifting gliding smooth flowing",
}

_APPEARANCE_RANGES = {
    "body_radius": (0.15, 0.50),
    "head_scale":  (0.60, 1.50),
    "limb_length": (0.20, 0.90),
    "tail_length": (0.10, 0.80),
    "ear_pointy":  (0.00, 1.00),
}


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class AppearanceParams:
    body_radius:   float = 0.30
    head_scale:    float = 1.00
    limb_length:   float = 0.50
    tail_length:   float = 0.40
    ear_pointy:    float = 0.50
    primary_hue:   float = 0.0
    secondary_hue: float = 180.0
    tail_hue:      float = 90.0
    limb_hue:      float = 270.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def lerp_shape(cls, a: "AppearanceParams", b: "AppearanceParams", t: float = 0.5) -> "AppearanceParams":
        def _l(x: float, y: float) -> float:
            return x + (y - x) * t
        return cls(
            body_radius   = _l(a.body_radius, b.body_radius),
            head_scale    = _l(a.head_scale,  b.head_scale),
            limb_length   = _l(a.limb_length, b.limb_length),
            tail_length   = _l(a.tail_length, b.tail_length),
            ear_pointy    = _l(a.ear_pointy,  b.ear_pointy),
        )


def _extract_hue(text: str, probes: dict[float, str], embedder: Embedder) -> float:
    text_vec = embedder.embed_single(text)
    best_hue, best_sim = 0.0, -1.0
    for hue, probe in probes.items():
        probe_vec = embedder.embed_single(probe)
        sim = _cosine_sim(text_vec, probe_vec)
        if sim > best_sim:
            best_sim = sim
            best_hue = hue
    return best_hue


def extract_appearance(genes: dict[str, str], embedder: Embedder) -> AppearanceParams:
    """Extract numeric appearance params from genes."""
    personality  = genes.get("personality", "")
    social_style = genes.get("social_style", "")
    reaction     = genes.get("reaction_pattern", "")
    movement     = genes.get("movement_style", "")

    # Use personality + reaction for body shape estimation
    shape_text = " ".join(filter(None, [personality, reaction]))
    params: dict[str, float] = {}

    if shape_text:
        shape_vec = embedder.embed_single(shape_text)
        for param, probe_text in APPEARANCE_PROBES.items():
            probe_vec = embedder.embed_single(probe_text)
            raw = _cosine_sim(shape_vec, probe_vec)
            normalized = max(0.0, min(1.0, (raw - 0.40) / 0.35))
            lo, hi = _APPEARANCE_RANGES[param]
            params[param] = lo + normalized * (hi - lo)
    else:
        for param, (lo, hi) in _APPEARANCE_RANGES.items():
            params[param] = (lo + hi) / 2.0

    primary_hue   = _extract_hue(personality,  PERSONALITY_COLOR_PROBES, embedder) if personality  else 60.0
    secondary_hue = _extract_hue(social_style, SOCIAL_COLOR_PROBES,      embedder) if social_style else (primary_hue + 150) % 360
    tail_hue      = _extract_hue(reaction,     REACTION_COLOR_PROBES,    embedder) if reaction     else 90.0
    limb_hue      = _extract_hue(movement,     MOVEMENT_COLOR_PROBES,    embedder) if movement     else 270.0

    return AppearanceParams(
        body_radius   = params.get("body_radius",  0.30),
        head_scale    = params.get("head_scale",   1.00),
        limb_length   = params.get("limb_length",  0.50),
        tail_length   = params.get("tail_length",  0.40),
        ear_pointy    = params.get("ear_pointy",   0.50),
        primary_hue   = primary_hue,
        secondary_hue = secondary_hue,
        tail_hue      = tail_hue,
        limb_hue      = limb_hue,
    )


# ---------------------------------------------------------------------------
# Skill extraction
# ---------------------------------------------------------------------------

SKILL_PROBES = {
    "can_roll":     "tumbles rolls spins rotates body along ground",
    "tight_circle": "runs in tight circles circular spiral patterns",
    "bounce_gait":  "bouncy springy energetic hopping movement",
    "can_sneak":    "slow quiet stealthy creeping cautious movement",
    "speed":        "fast agile sprint rapid movement quick",
}

_SKILL_THRESHOLD = 0.60


@dataclass
class SkillSet:
    can_roll:     bool  = False
    tight_circle: bool  = False
    bounce_gait:  bool  = False
    can_sneak:    bool  = False
    speed_base:   float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


def extract_skills(genes: dict[str, str], embedder: Embedder) -> SkillSet:
    movement = genes.get("movement_style", "")
    if not movement:
        return SkillSet()

    move_vec = embedder.embed_single(movement)
    scores: dict[str, float] = {}
    for skill, probe_text in SKILL_PROBES.items():
        probe_vec = embedder.embed_single(probe_text)
        raw = _cosine_sim(move_vec, probe_vec)
        scores[skill] = max(0.0, min(1.0, (raw - 0.30) / 0.50))

    speed_score = scores.get("speed", 0.5)
    speed_base  = 0.5 + speed_score * 1.5

    return SkillSet(
        can_roll     = scores.get("can_roll",     0.0) > _SKILL_THRESHOLD,
        tight_circle = scores.get("tight_circle", 0.0) > _SKILL_THRESHOLD,
        bounce_gait  = scores.get("bounce_gait",  0.0) > _SKILL_THRESHOLD,
        can_sneak    = scores.get("can_sneak",     0.0) > _SKILL_THRESHOLD,
        speed_base   = round(speed_base, 2),
    )


# ---------------------------------------------------------------------------
# EntityStats extraction (from spatial evolution)
# ---------------------------------------------------------------------------

STAT_PROBES = {
    "speed": "Moves with exceptional speed and agility across terrain",
    "damage": "Delivers devastating attacks that cause severe injury",
    "defense": "Withstands physical attacks with tough hide and endurance",
    "attractiveness": "Displays vibrant signals of genetic fitness and beauty",
    "breed_drive": "Strongly motivated to find mates and reproduce frequently",
    "food_finding": "Expert at locating, tracking, and harvesting food efficiently",
    "climate_resist": "Survives extreme temperatures, storms, and harsh conditions",
    "aggression": "Initiates confrontation, fights rivals, attacks on sight",
    "sociability": "Cooperates, forms alliances, shares resources with others",
    "exploration_drive": "Compelled to scout unknown territory and discover new areas",
}


@dataclass
class EntityStats:
    speed: float = 0.5
    damage: float = 0.3
    defense: float = 0.3
    attractiveness: float = 0.3
    breed_drive: float = 0.3
    food_finding: float = 0.5
    climate_resist: float = 0.3
    aggression: float = 0.3
    sociability: float = 0.3
    exploration_drive: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def extract_stats(genes: dict[str, str], embedder: Embedder) -> EntityStats:
    """Convert text genes to numeric stats via embedding similarity."""
    if not genes:
        return EntityStats()

    gene_vecs = [embedder.embed_single(t) for t in genes.values()]
    probe_vecs = {
        name: embedder.embed_single(text)
        for name, text in STAT_PROBES.items()
    }

    stat_vals: dict[str, float] = {}
    for stat_name, probe_vec in probe_vecs.items():
        sims = [_cosine_sim(gv, probe_vec) for gv in gene_vecs]
        raw = max(sims) if sims else 0.0
        normalized = (raw - 0.40) / 0.35
        stat_vals[stat_name] = max(0.05, min(1.0, normalized))

    return EntityStats(**stat_vals)


# ---------------------------------------------------------------------------
# Behavior profile (BEAR retrieval per situation)
# ---------------------------------------------------------------------------

@dataclass
class Situation:
    name: str
    query: str
    tags: list[str]
    primary_cat: str  # gene category that must match


# 25 situations — each maps to a primary gene category via required_tags.
# Multiple situations per category create a higher-dimensional behavior
# profile that resists cosine-similarity saturation in 9-dim space.
SITUATIONS: list[Situation] = [
    # --- foraging (3) ---
    Situation("food_seeking",
              "I'm hungry and need to find food to survive",
              ["food", "hunger", "foraging", "survival"],
              "foraging"),
    Situation("food_competition",
              "Another creature is heading for the same food I spotted",
              ["food", "competition", "foraging", "nearby"],
              "foraging"),
    Situation("food_scarcity",
              "Food is extremely scarce and I haven't eaten in a long time",
              ["food", "hunger", "survival", "starvation"],
              "foraging"),
    # --- predator_defense (3) ---
    Situation("combat",
              "Another creature is nearby and could be a threat",
              ["combat", "threat", "defense", "intruder"],
              "predator_defense"),
    Situation("group_defense",
              "A predator is attacking a nearby ally",
              ["combat", "social", "predator", "defense"],
              "predator_defense"),
    Situation("threat_assessment",
              "An unknown creature is approaching and I cannot tell if it is friendly",
              ["threat", "unknown", "caution", "defense"],
              "predator_defense"),
    # --- social_style (3) ---
    Situation("breeding",
              "A potential mate is nearby, should I reproduce",
              ["breed", "social", "nearby"],
              "social_style"),
    Situation("social",
              "A group of creatures is nearby",
              ["social", "nearby", "other"],
              "social_style"),
    Situation("cooperation",
              "Another creature is struggling with a task nearby and might need help",
              ["social", "cooperation", "nearby", "help"],
              "social_style"),
    Situation("herding",
              "A predator is nearby and I should group up with others for safety",
              ["predator", "social", "danger", "nearby"],
              "social_style"),
    # --- climate_survival (3) ---
    Situation("survival",
              "Weather is harsh and conditions are dangerous",
              ["weather", "climate", "survival", "harsh"],
              "climate_survival"),
    Situation("shelter_seeking",
              "A storm is approaching and I need to find shelter quickly",
              ["weather", "shelter", "survival", "storm"],
              "climate_survival"),
    Situation("heat_endurance",
              "The temperature is rising to extreme levels and becoming unbearable",
              ["weather", "heat", "survival", "climate"],
              "climate_survival"),
    # --- territorial (2) ---
    Situation("territory",
              "Something has entered my area",
              ["territory", "intruder", "combat", "space"],
              "territorial"),
    Situation("resource_guarding",
              "Another creature is approaching a resource I have been protecting",
              ["territory", "food", "guard", "defense"],
              "territorial"),
    # --- personality (3) ---
    Situation("exploration",
              "I'm wandering and discovering new things in unfamiliar territory",
              ["idle", "wander", "general"],
              "personality"),
    Situation("curiosity",
              "I see something unusual and glowing in the distance",
              ["explore", "curiosity", "unknown", "wander"],
              "personality"),
    Situation("rest_decision",
              "I have been active for a while and need to decide whether to keep going or rest",
              ["idle", "rest", "energy", "general"],
              "personality"),
    # --- stealth (2) ---
    Situation("stealth",
              "I need to hide and avoid being detected by a nearby predator",
              ["stealth", "hide", "camouflage", "predator"],
              "stealth"),
    Situation("ambush",
              "I want to approach prey without being noticed",
              ["stealth", "hunt", "camouflage", "food"],
              "stealth"),
    # --- sensory (2) ---
    Situation("detection",
              "I need to scan my surroundings for approaching threats or food sources",
              ["sense", "detect", "awareness", "predator", "danger"],
              "sensory"),
    Situation("tracking",
              "I found tracks or traces of another creature and need to follow them",
              ["sense", "track", "awareness", "hunt"],
              "sensory"),
    # --- movement_style (2) — previously unmapped ---
    Situation("navigation",
              "I need to move and walk across the ground to reach somewhere, choosing my pace and stride pattern",
              ["movement", "terrain", "travel", "navigate"],
              "movement_style"),
    Situation("pacing",
              "I am walking and moving around with no particular urgency, just covering ground at my natural pace",
              ["movement", "idle", "travel", "wander"],
              "movement_style"),
    # --- reaction_pattern (2) — previously unmapped ---
    Situation("surprise_reaction",
              "Something unexpected just happened and I need to react and respond immediately",
              ["surprise", "threat", "reaction", "startle"],
              "reaction_pattern"),
    Situation("noise_response",
              "A sudden stimulus demands my immediate response and first instinct reaction",
              ["sound", "alert", "reaction", "awareness"],
              "reaction_pattern"),
]

SITUATION_NAMES = [s.name for s in SITUATIONS]


@dataclass
class SituationResult:
    strength:      float
    gene_category: str
    gene_text:     str
    similarity:    float


class NullBehaviorProfile:
    """Uniform 0.3 profile used for BEAR-Off ablation condition.
    All situations return the same neutral score so tick() has no
    BEAR-driven guidance."""
    def strength(self, situation_name: str) -> float:
        return 0.3
    def to_dict(self) -> dict:
        return {}


@dataclass
class BehaviorProfile:
    situations: dict[str, SituationResult]

    def strength(self, situation_name: str) -> float:
        r = self.situations.get(situation_name)
        return r.strength if r else 0.3

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "strength":      round(r.strength, 3),
                "gene_category": r.gene_category,
                "gene_text":     r.gene_text[:80],
                "similarity":    round(r.similarity, 3),
            }
            for name, r in self.situations.items()
        }


def compute_behavior_profile(
    corpus: Corpus,
    config: Config,
    shared_embedder: "Embedder | None" = None,
) -> BehaviorProfile:
    """Run BEAR retrieval for each situation against the creature's corpus."""
    retriever = Retriever(corpus, config=config)
    if shared_embedder is not None:
        retriever._embedder = shared_embedder
    retriever.build_index()

    results: dict[str, SituationResult] = {}
    for sit in SITUATIONS:
        ctx = Context(tags=sit.tags + [f"cat:{sit.primary_cat}"])
        scored = retriever.retrieve(query=sit.query, context=ctx, top_k=3, threshold=0.3)
        if scored:
            top = scored[0]
            cat = top.instruction.metadata.get("gene_category", "unknown")
            results[sit.name] = SituationResult(
                strength      = top.final_score,
                gene_category = cat,
                gene_text     = top.instruction.content,
                similarity    = top.similarity,
            )
        else:
            results[sit.name] = SituationResult(
                strength=0.0, gene_category="none", gene_text="", similarity=0.0,
            )

    return BehaviorProfile(situations=results)


# ---------------------------------------------------------------------------
# Build BEAR corpus from all 8 gene categories
# ---------------------------------------------------------------------------

_CORPUS_TEMPLATES: dict[str, list[tuple[list[str], str]]] = {
    "personality": [
        (["idle", "wander", "general"],
         "Relaxed and alone — disposition: {gene_text} "
         "Use [!wander] to explore or [!approach(item=food)] if hungry. "
         "[!thought(just taking it easy)]"),
        (["mood_excited", "mood_happy", "general"],
         "Feeling excited — personality: {gene_text} "
         "Use [!sprint] or [!roll] to express this energy. [!mood(excited)]"),
        (["mood_cautious", "mood_annoyed", "stimulus"],
         "Uncomfortable or wary — reaction: {gene_text} "
         "Use [!sneak] and [!mood(cautious)] to stay alert."),
        (["skill_rolling", "idle", "general"],
         "Feeling playful and nimble — personality: {gene_text} "
         "Use [!roll] to tumble around. [!mood(playful)]"),
        (["skill_bounce", "mood_happy", "general"],
         "Bouncy and cheerful — personality: {gene_text} "
         "Use [!circle] to bounce in a happy pattern. [!mood(happy)]"),
    ],
    "social_style": [
        (["nearby", "social", "other"],
         "Another creature is close — social instinct: {gene_text} "
         "Use [!approach(nearest)] to interact or [!wander] to keep distance."),
        (["greeting", "social"],
         "First time meeting someone — social style: {gene_text} "
         "Use [!approach(nearest)] to greet them. [!mood(happy)] "
         "[!thought(hello there)]"),
        (["social", "mood_playful"],
         "A playful social moment — style: {gene_text} "
         "Use [!approach(nearest)] and [!mood(playful)] to play together. [!roll]"),
    ],
    "mating": [
        (["breed"],
         "{gene_text}",
         ["breed"]),
    ],
    "reaction_pattern": [
        (["stimulus", "idle", "nearby"],
         "Something caught attention — reaction: {gene_text} "
         "Use [!approach(nearest)] to investigate or [!sneak] to observe cautiously. "
         "[!thought(what is that)]"),
        (["idle", "wander"],
         "No stimulus — default pattern: {gene_text} "
         "Use [!wander] to move around the environment."),
        (["mood_cautious", "stimulus"],
         "Uncertain stimulus — response: {gene_text} "
         "Use [!sneak] and [!mood(cautious)] to assess carefully."),
    ],
    "foraging": [
        (["food", "hunger", "survival"],
         "Hungry — foraging strategy: {gene_text} "
         "Use [!approach(item=food)] to find food. [!mood(cautious)] "
         "[!thought(need to eat)]"),
        (["food", "foraging", "idle"],
         "Could gather food — foraging instinct: {gene_text} "
         "Use [!approach(item=food)] to seek food or [!wander] to forage."),
        (["food_nearby", "hunger"],
         "Food spotted while hungry — foraging drive: {gene_text} "
         "Use [!approach(item=food)] [!sprint] to grab it quickly."),
        (["survival", "hunger", "harsh"],
         "Survival critical — foraging: {gene_text} "
         "Use [!approach(item=food)] [!sprint] to find food urgently. "
         "[!thought(starving)]"),
    ],
    "predator_defense": [
        (["predator", "danger", "flee"],
         "Predator detected — defense strategy: {gene_text} "
         "React with [!flee] to escape. "
         "[!mood(cautious)] [!thought(run)]",
         ["flee"]),
        (["predator", "danger", "rally"],
         "Predator attack — strength in numbers: {gene_text} "
         "Use [!approach(nearest)] to group up, then [!rally] to fight back together. "
         "[!mood(excited)] [!thought(stand together)]",
         ["rally"]),
        (["defense", "survival", "danger"],
         "Need to protect self — defense instinct: {gene_text} "
         "Use [!flee] to escape danger or [!rally] to defend with nearby allies."),
    ],
    "climate_survival": [
        (["weather", "climate", "survival"],
         "Weather is harsh — climate survival: {gene_text} "
         "Use [!approach(item=tree)] to find shelter or [!sneak] to conserve energy."),
        (["harsh", "survival", "climate"],
         "Extreme conditions — adaptation: {gene_text} "
         "Use [!approach(item=tree)] for shelter. [!mood(cautious)] "
         "[!thought(need cover)]"),
        (["weather", "idle"],
         "Weather awareness — climate instinct: {gene_text} "
         "Use [!wander] to find better conditions or [!approach(item=tree)] for cover."),
    ],
    "territorial": [
        (["enraged", "aggression", "nearby"],
         "Overcome with rage — territorial fury: {gene_text} "
         "Use [!challenge(nearest)] to confront the intruder. [!mood(annoyed)]"),
        (["territory", "idle"],
         "Surveying territory — behavior: {gene_text} "
         "Use [!wander] to patrol the area. [!mood(cautious)]"),
        (["combat", "nearby", "aggression"],
         "Rival nearby — territorial: {gene_text} "
         "Use [!challenge(nearest)] to confront or [!sneak] to withdraw."),
    ],
    "stealth": [
        (["stealth", "predator", "danger"],
         "Need to avoid detection — stealth capability: {gene_text} "
         "Use [!sneak] and [!mood(cautious)] to hide from threats."),
        (["stealth", "idle"],
         "Trying to remain unseen — concealment instinct: {gene_text} "
         "Use [!sneak] to stay hidden. [!thought(stay quiet)]"),
        (["skill_sneak", "predator", "danger"],
         "Predator nearby, must hide — stealth response: {gene_text} "
         "Use [!sneak] to avoid being detected. [!mood(cautious)]"),
    ],
    "sensory": [
        (["sense", "idle", "general"],
         "Scanning environment — senses: {gene_text} "
         "Use [!wander] to scan the area. [!mood(cautious)]"),
        (["sense", "predator", "danger"],
         "Watching for predators — detection: {gene_text} "
         "Use [!sneak] to stay alert or [!flee] if threat detected."),
        (["idle", "wander", "general"],
         "Environmental awareness — senses: {gene_text} "
         "Use [!wander] to explore and observe surroundings. "
         "[!thought(looking around)]"),
    ],
}


def build_corpus(
    name: str,
    genes: dict[str, str],
    dominances: dict[str, float] | None = None,
) -> Corpus:
    """Build BEAR Corpus with situational instructions per gene category.

    Each template entry is either:
        (scope_tags, content_template)
        (scope_tags, content_template, required_tags)
    When required_tags is provided, ALL listed tags must be present in the
    retrieval context for the instruction to be retrieved (AND logic).

    *dominances* maps gene category -> per-allele dominance score (0.0..1.0,
    higher is more dominant). Used by ``express()`` for DOMINANT loci to
    decide which allele wins in a heterozygote pairing. If None, all
    instructions get a default score of 1.0 (which makes legacy untagged
    corpora behave as homozygous-equivalent under the score-max rule).
    """
    corpus = Corpus()
    for cat in BEHAVIOR_CATEGORIES:
        text = genes.get(cat)
        if not text:
            continue
        templates = _CORPUS_TEMPLATES.get(cat, [])
        score = (dominances or {}).get(cat, 1.0)
        for idx, entry in enumerate(templates):
            scope_tags = entry[0]
            content_template = entry[1]
            req_tags = entry[2] if len(entry) > 2 else []
            prompt = content_template.format(gene_text=text)
            corpus.add(Instruction(
                id       = f"{name}-{cat}-{idx}",
                type     = InstructionType.DIRECTIVE,
                priority = _CATEGORY_PRIORITY.get(cat, 60),
                content  = text,  # raw gene text — creature genetics drive similarity
                scope    = ScopeCondition(tags=scope_tags, required_tags=req_tags),
                tags     = [name, cat, f"cat:{cat}"] + scope_tags,
                metadata = {"gene_category": cat, "situation_idx": idx,
                            "prompt": prompt, "dominance": score},
            ))
    return corpus


def random_founder_dominances(
    rng: random.Random,
    categories: list[str] | None = None,
) -> dict[str, float]:
    """Draw per-category dominance scores for a founder NPC.

    Founders sample uniformly from [0, 1]: roughly half are dominant-leaning,
    half recessive-leaning at any given locus. This produces biologically
    realistic gen-0 populations where archetype alleles aren't all clustered
    at the top of the dominance hierarchy.
    """
    cats = categories if categories is not None else GENE_CATEGORIES
    return {cat: rng.random() for cat in cats}


def random_mutation_dominance(
    rng: random.Random,
    *,
    alpha: float = 1.0,
    beta: float = 4.0,
) -> float:
    """Draw a dominance score for a newly-mutated allele.

    Defaults to ``Beta(1, 4)``: heavily biased recessive (mean ~0.2,
    most mass below 0.4), modeling the biological observation that most
    de novo mutations are recessive deleterious. Tunable via *alpha*
    and *beta*; ``alpha=beta=1`` recovers a uniform distribution.
    """
    return rng.betavariate(alpha, beta)


def random_drift_dominance(rng: random.Random) -> float:
    """Draw a dominance score for a drift (fully novel) allele.

    Drift produces content unrelated to any parent allele. Such variants
    are biologically the most often recessive; we sample from
    ``Beta(1, 6)`` (mean ~0.14, strongly skewed toward 0).
    """
    return rng.betavariate(1.0, 6.0)


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You design behavioral traits and survival strategies for whimsical NPCs in a "
    "creature evolution simulation. Be creative, specific, and varied."
)

_INIT_PROMPT = """\
Generate traits for a whimsical NPC creature — one trait per category.
Each trait should be a specific, vivid description in 1-3 sentences.
Different creatures should feel genuinely different from each other.

Categories:
{category_list}

Output a JSON object mapping category name to trait string.
Example: {{"personality": "...", "foraging": "...", ...}}

Output ONLY the JSON object, nothing else."""

_BLEND_PROMPT = """\
Two parent creatures are producing an offspring. Blend their {category} traits.

Parent A ({category}): "{content_a}"
Parent B ({category}): "{content_b}"

Generate one trait (1-3 sentences) combining elements from both parents.

Output ONLY the trait text, nothing else."""

_MUTATE_PROMPT = """\
A creature has this {category} trait:
"{content}"

Generate a mutated version — a plausible variation with a different twist.
Keep it to 1-3 sentences.

Output ONLY the trait text, nothing else."""

_DRIFT_PROMPT = """\
Generate a completely new {category} trait for an NPC creature.
Make it creative, unusual, and specific. Avoid generic descriptions.

Output ONLY the trait text, nothing else."""

_NAME_PROMPT = """\
Give this creature a cute, short single-word name that matches its personality.
Good examples: Wobble, Pip, Mochi, Biscuit, Zippy, Blaze, Wisp, Sage, Fang, Spark.

Personality: "{personality}"

Output ONLY the name, one word, nothing else."""

# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

_REASONING_PAT = re.compile(
    r"(?i)^(okay|let me|let's|first|hmm|so\b|the key|wait|ah|"
    r"i need|since|well|alright|now|right|maybe|perhaps|the user|"
    r"generate a|here is|here's|sure|of course|output|yes[,.]|"
    r"that's a|this is a|i'll|i will)"
)

_PROMPT_ECHO = re.compile(
    r"(?i)(output only|json object|example format|behavioral category|categories:)"
)


def _clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip().strip('"').strip()
    if _REASONING_PAT.match(text):
        quoted = re.findall(r'"([^"]{20,})"', text)
        if quoted:
            text = quoted[0]
        else:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            short = [p for p in paragraphs if len(p) < 400]
            if short:
                text = short[-1]
        if _REASONING_PAT.match(text):
            return ""
    if len(text) > 400:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) > 3:
            text = " ".join(sentences[:3])
    return text.strip().strip('"').strip()


def _is_valid(text: str) -> bool:
    if len(text) < 10 or len(text) > 400:
        return False
    if _PROMPT_ECHO.search(text):
        return False
    if _REASONING_PAT.match(text):
        return False
    return True


def _parse_gene_dict(raw: str) -> dict[str, str]:
    text = raw.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                result: dict[str, str] = {}
                for cat in GENE_CATEGORIES:
                    val = obj.get(cat, "")
                    if isinstance(val, str) and _is_valid(val.strip()):
                        result[cat] = val.strip()
                return result
        except json.JSONDecodeError:
            pass
    return {}


_FALLBACK_GENES: dict[str, str] = {
    "personality":      "Curious and gentle, always looking around with wide eyes.",
    "social_style":     "Friendly to strangers, enjoys being in groups.",
    "movement_style":   "Walks at a steady pace, occasionally skipping.",
    "reaction_pattern": "Investigates new things slowly and carefully.",
    "foraging":         "Patiently searches the ground for food, sniffs the air for scents.",
    "predator_defense": "Freezes when sensing danger, then bolts for the nearest cover.",
    "climate_survival": "Huddles into a compact ball when weather turns harsh.",
    "territorial":      "Marks territory boundaries and patrols them regularly.",
    "stealth":          "Stays low and quiet when threatened, blending into surroundings as best it can.",
    "sensory":          "Average awareness of surroundings, notices obvious threats at moderate distance.",
    "mating":           "Moderate interest in reproducing [!approach(nearest)] [!breed(nearest)]",
}

# ---------------------------------------------------------------------------
# Async gene + name generation
# ---------------------------------------------------------------------------


_GENE_BANK: list[dict[str, str]] = []  # loaded lazily to avoid circular import

def _get_gene_bank() -> list[dict[str, str]]:
    """Load the gene bank lazily from the eval harness."""
    global _GENE_BANK
    if not _GENE_BANK:
        try:
            from examples.evolutionary_ecosystem.eval.harness import GENE_BANK
            _GENE_BANK = list(GENE_BANK)
            logger.info("Gene bank loaded: %d archetypes", len(_GENE_BANK))
        except Exception as e:
            logger.warning("Gene bank not available (%s) — will use LLM generation", e)
    return _GENE_BANK

# Mating genes to supplement the gene bank (which predates the mating category)
_MATING_GENES: list[str] = [
    "Very eager to reproduce [!approach(nearest)] [!breed(nearest)] [!mood(happy)]",
    "Eager to reproduce [!approach(nearest)] [!breed(nearest)] [!mood(happy)]",
    "Moderate interest in reproducing [!approach(nearest)] [!breed(nearest)]",
    "Low interest in reproducing [!approach(nearest)] [!breed(nearest)]",
    "Rarely interested in reproducing [!approach(nearest)]",
]


async def generate_npc_genes(llm: LLM, rng: random.Random) -> dict[str, str]:
    """Generate initial genes — from gene bank if available, else via LLM."""
    gene_bank = _get_gene_bank()
    if gene_bank:
        # Pick a random archetype and mix categories across archetypes
        base = rng.choice(gene_bank)
        result: dict[str, str] = {}
        for cat in GENE_CATEGORIES:
            if cat == "mating":
                result[cat] = rng.choice(_MATING_GENES)
            elif cat in base:
                result[cat] = base[cat]
            else:
                # Mix from a different random archetype
                donor = rng.choice(gene_bank)
                result[cat] = donor.get(cat, _FALLBACK_GENES.get(cat, ""))
        return result

    # Fallback: LLM generation
    cat_list = "\n".join(
        f"- {cat}: {_CATEGORY_DESCRIPTIONS[cat]}"
        for cat in GENE_CATEGORIES
    )
    resp = await llm.generate(
        system=_SYSTEM,
        user=_INIT_PROMPT.format(category_list=cat_list),
        temperature=0.9,
        max_tokens=2048,
    )
    genes = _parse_gene_dict(resp.content)

    result = {}
    for cat in GENE_CATEGORIES:
        if cat == "mating":
            result[cat] = rng.choice(_MATING_GENES)
        else:
            result[cat] = genes.get(cat) or _FALLBACK_GENES[cat]
    return result


# Curated creature names — varied, memorable, no LLM needed
_CREATURE_NAMES = [
    "Blaze", "Pip", "Fern", "Rook", "Wisp", "Dart", "Moss", "Flint",
    "Ziggy", "Sage", "Bolt", "Dew", "Crag", "Fizz", "Reed", "Tumble",
    "Spark", "Nook", "Dash", "Clover", "Fang", "Mist", "Boing", "Cedar",
    "Honey", "Pebble", "Whirl", "Stone", "Sunny", "Drift", "Maple", "Bounce",
    "Briar", "Thorn", "Gust", "Puddle", "Flicker", "Burr", "Ember", "Gravel",
    "Thistle", "Brook", "Cinder", "Dune", "Frond", "Gale", "Hazel", "Ivy",
    "Juniper", "Kelp", "Lichen", "Marrow", "Nettle", "Ochre", "Prism", "Quill",
    "Sedge", "Tangle", "Umbra", "Vetch", "Wren", "Yarrow", "Zephyr", "Acorn",
]


async def generate_creature_name(llm: LLM, genes: dict[str, str], rng: random.Random | None = None) -> str:
    personality = genes.get("personality", "curious and friendly")
    try:
        resp = await llm.generate(
            system=_SYSTEM,
            user=_NAME_PROMPT.format(personality=personality[:200]),
            temperature=1.1,
            max_tokens=32,
        )
        raw = resp.content.strip()
        _m = re.search(r'\{[^{}]*"name"\s*:\s*"([^"]{2,20})"[^{}]*\}', raw)
        if _m:
            name = _m.group(1).strip()
            if name.isalpha():
                return name.capitalize()
        words = re.findall(r'\b[A-Za-z]{2,20}\b', raw)
        _SKIP = {"thinking","think","the","name","output","json","object",
                 "personality","give","cute","short","good","bad","example"}
        words = [w for w in reversed(words) if w.lower() not in _SKIP]
        name = words[0] if words else ""
        if 2 <= len(name) <= 20 and name.isalpha():
            return name.capitalize()
    except Exception:
        pass
    # Fallback to curated list if LLM fails
    _rng = rng or random.Random()
    return _rng.choice(_CREATURE_NAMES)

async def blend_gene(llm: LLM, category: str, content_a: str, content_b: str) -> str:
    resp = await llm.generate(
        system=_SYSTEM,
        user=_BLEND_PROMPT.format(category=category, content_a=content_a, content_b=content_b),
        temperature=0.8,
        max_tokens=512,
    )
    result = _clean(resp.content)
    return result if result else content_a


async def mutate_gene(llm: LLM, category: str, content: str) -> str:
    resp = await llm.generate(
        system=_SYSTEM,
        user=_MUTATE_PROMPT.format(category=category, content=content),
        temperature=0.9,
        max_tokens=512,
    )
    result = _clean(resp.content)
    return result if result else content


async def drift_gene(llm: LLM, category: str) -> str:
    resp = await llm.generate(
        system=_SYSTEM,
        user=_DRIFT_PROMPT.format(category=category),
        temperature=1.0,
        max_tokens=512,
    )
    result = _clean(resp.content)
    return result if result else _FALLBACK_GENES.get(category, "")


# ---------------------------------------------------------------------------
# Breeding dataclasses + async pipeline
# ---------------------------------------------------------------------------

@dataclass
class BreedRequest:
    parent_a_genes:    dict[str, str]
    parent_b_genes:    dict[str, str]
    parent_a_name:     str
    parent_b_name:     str
    parent_a_corpus:   Corpus
    parent_b_corpus:   Corpus
    parent_a_appear:   AppearanceParams
    parent_b_appear:   AppearanceParams
    parent_a_fitness:  float
    parent_b_fitness:  float
    child_name:        str
    child_id:          str
    spawn_x:           float
    spawn_y:           float
    generation:        int
    mutation_rate:     float = 0.15
    recombination:     str = "locus"    # "locus" | "blend" | "splice"
    ploidy:            str = "haploid"  # "haploid" | "diploid_dominant" | "diploid_codominant"


@dataclass
class BreedResult:
    child_id:       str
    child_name:     str
    genes:          dict[str, str]
    corpus:         Corpus
    appearance:     AppearanceParams
    skills:         SkillSet
    stats:          EntityStats
    behavior:       BehaviorProfile
    spawn_x:        float
    spawn_y:        float
    generation:     int
    parent_a_name:  str
    parent_b_name:  str


def _make_locus_registry(ploidy: str) -> LocusRegistry:
    """Build a LocusRegistry with the appropriate dominance model."""
    dom_map = {
        "haploid":              Dominance.HAPLOID,
        "diploid_dominant":     Dominance.DOMINANT,
        "diploid_codominant":   Dominance.CODOMINANT,
    }
    dom = dom_map.get(ploidy, Dominance.HAPLOID)
    return LocusRegistry(loci=[
        GeneLocus(name=cat, position=i, dominance=dom)
        for i, cat in enumerate(GENE_CATEGORIES)
    ])


async def _mutate_corpus(
    corpus: Corpus,
    locus_key: str,
    mutation_rate: float,
    llm: LLM,
    rng: random.Random,
) -> Corpus:
    """Apply per-locus mutations to a (possibly diploid) corpus.

    For each locus, with probability *mutation_rate*, mutate one randomly
    chosen allele. All instructions sharing the same (locus, allele)
    receive the same mutated content so the diploid genotype stays
    self-consistent across template variants. Mutated alleles also receive
    a fresh dominance score (Beta-distributed, recessive-biased), so
    novel alleles enter the gene pool more often as recessive carriers
    than as dominant variants.
    """
    by_pair: dict[tuple[str, str | None], list[Instruction]] = {}
    other: list[Instruction] = []
    for inst in corpus:
        loc = inst.metadata.get(locus_key)
        if loc is None:
            other.append(inst)
            continue
        allele = inst.metadata.get("allele")
        by_pair.setdefault((loc, allele), []).append(inst)

    by_locus: dict[str, dict[str | None, list[Instruction]]] = {}
    for (loc, allele), insts in by_pair.items():
        by_locus.setdefault(loc, {})[allele] = insts

    mutated_text: dict[tuple[str, str | None], tuple[str, float]] = {}
    for loc, allele_map in by_locus.items():
        if rng.random() >= mutation_rate:
            continue
        chosen_allele = rng.choice(list(allele_map.keys()))
        seed_content = allele_map[chosen_allele][0].content
        if rng.random() < 0.3:
            new_text = await drift_gene(llm, loc)
            new_dominance = random_drift_dominance(rng)
        else:
            new_text = await mutate_gene(llm, loc, seed_content)
            new_dominance = random_mutation_dominance(rng)
        mutated_text[(loc, chosen_allele)] = (new_text, new_dominance)

    out = Corpus()
    for inst in other:
        out.add(inst)
    for (loc, allele), insts in by_pair.items():
        update = mutated_text.get((loc, allele))
        if update is None:
            for inst in insts:
                out.add(inst)
        else:
            new_text, new_dominance = update
            for inst in insts:
                out.add(inst.model_copy(update={
                    "content": new_text,
                    "metadata": {**inst.metadata, "dominance": new_dominance},
                }))
    return out


def _extract_genes_dict(
    corpus: Corpus,
    is_diploid: bool,
    registry: LocusRegistry,
    locus_key: str = "gene_category",
) -> dict[str, str]:
    """Derive a haploid-view ``dict[locus -> text]`` summary from a corpus.

    For diploid corpora, uses the expressed phenotype (so dominance rules
    are honored). For haploid corpora, reads directly from the corpus.
    The corpus remains the genotype source of truth — this dict is only
    for prompt construction, appearance/skill extraction, and logging.
    """
    if is_diploid:
        insts_iter: list[Instruction] = list(express(corpus, registry, locus_key=locus_key))
    else:
        insts_iter = list(corpus)

    out: dict[str, str] = {}
    for inst in insts_iter:
        cat = inst.metadata.get(locus_key)
        if cat and cat not in out:
            out[cat] = inst.content
    return out


async def breed_offspring(
    request: BreedRequest,
    llm: LLM,
    embedder: Embedder,
    rng: random.Random,
    bear_config: Config | None = None,
) -> BreedResult:
    """Full breeding pipeline supporting three recombination methods and ploidy modes.

    Diploid inheritance (locus or splice recombination):
      1. Meiosis + fertilisation: bear.evolution.breed() segregates one
         allele per parent per locus and pairs the two gametes, producing
         a diploid child corpus with allele "a" from parent A's drawn
         allele and allele "b" from parent B's.
      2. Mutation: applied per (locus, allele) on the bred corpus.
         Mutated alleles receive a Beta-distributed (recessive-biased)
         dominance score so novel alleles enter the gene pool mostly as
         recessive carriers.
      3. The corpus is the genotype source of truth; child_genes is a
         haploid-view summary derived from the expressed phenotype.
    """
    recomb = request.recombination
    ploidy = request.ploidy
    is_diploid = ploidy.startswith("diploid")
    registry = _make_locus_registry(ploidy)

    if recomb == "blend":
        # LLM-mediated blending: produces one new text per locus, then
        # rebuild a (haploid-shaped) corpus. Diploid + blend is treated as
        # homozygous — both alleles would carry the same blended text, so
        # we just emit a single-allele corpus and let express() pass it
        # through unchanged.
        child_genes: dict[str, str] = {}
        blend_tasks = []
        blend_cats = []
        for cat in GENE_CATEGORIES:
            has_a = cat in request.parent_a_genes
            has_b = cat in request.parent_b_genes
            if has_a and has_b:
                blend_cats.append(cat)
                blend_tasks.append(blend_gene(llm, cat, request.parent_a_genes[cat], request.parent_b_genes[cat]))
            elif has_a:
                child_genes[cat] = request.parent_a_genes[cat]
            elif has_b:
                child_genes[cat] = request.parent_b_genes[cat]
            else:
                child_genes[cat] = _FALLBACK_GENES.get(cat, "")
        if blend_tasks:
            blend_results = await asyncio.gather(*blend_tasks)
            for cat, result in zip(blend_cats, blend_results):
                child_genes[cat] = result

        for cat in GENE_CATEGORIES:
            if rng.random() < request.mutation_rate:
                if rng.random() < 0.3:
                    child_genes[cat] = await drift_gene(llm, cat)
                else:
                    child_genes[cat] = await mutate_gene(llm, cat, child_genes.get(cat, ""))

        corpus = build_corpus(request.child_name, child_genes)

    else:
        # Locus-based or splice: use bear.evolution.breed. Bear handles
        # meiosis internally for diploid parents (segregates one allele
        # per parent per locus before pairing), so we just pass the
        # parent corpora through.
        if recomb == "splice":
            config = BreedingConfig(
                crossover_rate=0.5,
                seed=rng.randint(0, 2**31),
                scope_to_child=False,
            )
        else:
            config = BreedingConfig(
                crossover_rate=0.5,
                locus_key="gene_category",
                locus_registry=registry,
                crossover_method=CrossoverMethod.TAGGED,
                scope_to_child=False,
                seed=rng.randint(0, 2**31),
            )
        breed_result = breed(
            request.parent_a_corpus,
            request.parent_b_corpus,
            request.child_name,
            request.parent_a_name,
            request.parent_b_name,
            config=config,
        )

        # Trust bear's bred corpus and mutate it per (locus, allele).
        corpus = await _mutate_corpus(
            breed_result.child, "gene_category",
            request.mutation_rate, llm, rng,
        )

        # Derive haploid-view child_genes from the expressed phenotype
        # (used for prompts, appearance/skill/stat extraction, logging).
        child_genes = _extract_genes_dict(corpus, is_diploid, registry)
        for cat in GENE_CATEGORIES:
            if cat not in child_genes:
                child_genes[cat] = (request.parent_a_genes.get(cat)
                                    or request.parent_b_genes.get(cat)
                                    or _FALLBACK_GENES.get(cat, ""))

    # --- Step 4: Appearance, skills, stats, behavior ---
    total   = request.parent_a_fitness + request.parent_b_fitness
    t_blend = 0.5 if total == 0 else request.parent_b_fitness / total
    shape   = AppearanceParams.lerp_shape(request.parent_a_appear, request.parent_b_appear, t_blend)
    colors  = extract_appearance(child_genes, embedder)
    appearance = AppearanceParams(
        body_radius   = shape.body_radius,
        head_scale    = shape.head_scale,
        limb_length   = shape.limb_length,
        tail_length   = shape.tail_length,
        ear_pointy    = shape.ear_pointy,
        primary_hue   = colors.primary_hue,
        secondary_hue = colors.secondary_hue,
        tail_hue      = colors.tail_hue,
        limb_hue      = colors.limb_hue,
    )

    skills = extract_skills(child_genes, embedder)
    stats  = extract_stats(child_genes, embedder)

    # For diploid, compute behavior profile from expressed phenotype only
    if is_diploid and recomb != "blend":
        expressed_insts = express(corpus, registry, locus_key="gene_category")
        expressed_corpus = Corpus()
        for inst in expressed_insts:
            expressed_corpus.add(inst)
        cfg = bear_config or Config(embedding_backend=EmbeddingBackend.NUMPY, mandatory_tags=[])
        behavior = compute_behavior_profile(expressed_corpus, cfg, shared_embedder=embedder)
    else:
        cfg = bear_config or Config(embedding_backend=EmbeddingBackend.NUMPY, mandatory_tags=[])
        behavior = compute_behavior_profile(corpus, cfg, shared_embedder=embedder)

    return BreedResult(
        child_id      = request.child_id,
        child_name    = request.child_name,
        genes         = child_genes,
        corpus        = corpus,
        appearance    = appearance,
        skills        = skills,
        stats         = stats,
        behavior      = behavior,
        spawn_x       = request.spawn_x,
        spawn_y       = request.spawn_y,
        generation    = request.generation,
        parent_a_name = request.parent_a_name,
        parent_b_name = request.parent_b_name,
    )


def _build_diploid_corpus(
    name: str,
    child_genes: dict[str, str],
    parent_a_genes: dict[str, str],
    parent_b_genes: dict[str, str],
    registry: LocusRegistry,
) -> Corpus:
    """Build a diploid corpus carrying both parents' alleles at each locus."""
    corpus = Corpus()
    for cat in BEHAVIOR_CATEGORIES:
        allele_a = parent_a_genes.get(cat, "")
        allele_b = parent_b_genes.get(cat, "")
        if not allele_a and not allele_b:
            continue
        templates = _CORPUS_TEMPLATES.get(cat, [])
        for idx, entry in enumerate(templates):
            scope_tags = entry[0]
            content_template = entry[1]
            req_tags = entry[2] if len(entry) > 2 else []
            # Allele A
            if allele_a:
                prompt_a = content_template.format(gene_text=allele_a)
                corpus.add(Instruction(
                    id       = f"{name}-{cat}-{idx}-a",
                    type     = InstructionType.DIRECTIVE,
                    priority = _CATEGORY_PRIORITY.get(cat, 60),
                    content  = allele_a,
                    scope    = ScopeCondition(tags=scope_tags, required_tags=req_tags),
                    tags     = [name, cat, f"cat:{cat}"] + scope_tags,
                    metadata = {"gene_category": cat, "situation_idx": idx,
                                "prompt": prompt_a, "allele": "a"},
                ))
            # Allele B
            if allele_b and allele_b != allele_a:
                prompt_b = content_template.format(gene_text=allele_b)
                corpus.add(Instruction(
                    id       = f"{name}-{cat}-{idx}-b",
                    type     = InstructionType.DIRECTIVE,
                    priority = _CATEGORY_PRIORITY.get(cat, 60),
                    content  = allele_b,
                    scope    = ScopeCondition(tags=scope_tags, required_tags=req_tags),
                    tags     = [name, cat, f"cat:{cat}"] + scope_tags,
                    metadata = {"gene_category": cat, "situation_idx": idx,
                                "prompt": prompt_b, "allele": "b"},
                ))
    return corpus


async def breeding_worker(
    llm: LLM,
    embedder: Embedder,
    breed_queue: asyncio.Queue,
    birth_queue: asyncio.Queue,
    rng: random.Random,
    bear_config: Config | None = None,
):
    """Background task: pulls BreedRequests, runs pipeline, pushes BreedResults."""
    while True:
        request: BreedRequest = await breed_queue.get()
        try:
            result = await breed_offspring(request, llm, embedder, rng, bear_config)
            await birth_queue.put(result)
            logger.info(
                "Bred %s from %s x %s",
                result.child_name, result.parent_a_name, result.parent_b_name,
            )
        except Exception as e:
            logger.error("Breeding failed for %s: %s", request.child_name, e)
        finally:
            breed_queue.task_done()
