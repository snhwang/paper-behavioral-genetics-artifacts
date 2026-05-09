"""Shared evaluation harness for headless evolutionary ecosystem experiments.

Provides:
- Deterministic creature creation (no LLM — uses predefined gene banks)
- Deterministic breeding (text splicing instead of LLM blending)
- Headless simulation loop (tick-based, no async, no WebSocket)
- Population snapshot collection via PopulationTracker
"""

from __future__ import annotations

import asyncio
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Ensure repo root is on path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent))

# Silence noisy loggers before importing heavy modules
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")
import logging
for _ln in ("model2vec", "sentence_transformers", "safetensors", "httpx", "bear"):
    logging.getLogger(_ln).setLevel(logging.ERROR)

from bear import Config, Corpus, EmbeddingBackend
from bear.retriever import Embedder, Retriever
from bear.models import Dominance, GeneLocus, LocusRegistry
from bear.evolution import BreedingConfig, CrossoverMethod, breed as bear_breed, express

from examples.evolutionary_ecosystem.server.gene_engine import (
    GENE_CATEGORIES,
    AppearanceParams,
    BehaviorProfile,
    EntityStats,
    SkillSet,
    SituationResult,
    SITUATION_NAMES,
    build_corpus,
    compute_behavior_profile,
    extract_appearance,
    extract_skills,
    extract_stats,
    _FALLBACK_GENES,
)
from examples.evolutionary_ecosystem.server.sim import (
    Creature,
    FoodItem,
    Predator,
    World,
    WorldItem,
    tick,
    WORLD_W,
    WORLD_H,
    MAX_POPULATION,
    PREDATOR_SPAWN_INTERVAL,
)
from examples.evolutionary_ecosystem.server.epochs import (
    Epoch,
    EPOCHS,
    EPOCH_DURATION_TICKS,
)
from examples.evolutionary_ecosystem.server.stats import PopulationStats, PopulationTracker

# ---------------------------------------------------------------------------
# Predefined gene banks (diverse, hand-crafted for evaluation)
# ---------------------------------------------------------------------------

GENE_BANK: list[dict[str, str]] = [
    {   # Bold
        "personality": "Bold and fearless, charges headfirst into any situation with zero hesitation. Treats every encounter as an opportunity to prove dominance, never backing down even when outnumbered or outmatched.",
        "social_style": "Dominates social encounters through sheer force of presence. Demands attention and respect from everyone nearby, positioning itself at the center of any group and expecting others to defer.",
        "movement_style": "Sprints everywhere at full speed, never walks when running is an option. Powerful, ground-shaking strides that announce its approach long before it arrives.",
        "reaction_pattern": "Confronts anything new by puffing up and staring it down aggressively. First response to any stimulus is to charge toward it and assess at close range, never retreating to evaluate from safety.",
        "foraging": "Aggressively hunts food by charging toward any food source at full speed. Steals from others when hungry without hesitation, claims the best portions, and never shares willingly with competitors.",
        "predator_defense": "Stands ground against predators and counterattacks with fierce biting and clawing. Charges directly at threats rather than fleeing, relying on intimidation and raw aggression to drive them away. [!rally] [!challenge(nearest)]",
        "climate_survival": "Generates intense body heat through constant muscular activity. Thrives in cold conditions where its metabolic furnace provides warmth, but overheats easily in hot weather and needs shade frequently.",
        "territorial": "Viciously attacks any intruder that enters its territory. Patrols boundaries constantly with aggressive displays, leaving deep scratch marks and scent trails to warn others away. [!challenge(nearest)]",
        "stealth": "Makes no effort to hide. Moves loudly and visibly, relying on intimidation rather than concealment. Bright coloring and aggressive posturing announce its presence to everything nearby. [!sprint]",
        "sensory": "Poor sensory awareness due to constant aggression and noise-making. Relies on confrontation rather than early detection. Often fails to notice approaching threats until they are very close.",
        "mating": "Very eager to reproduce — actively pursues mates with bold displays and direct approaches. Rarely waits for the other party to initiate; treats breeding as another competition to win [!approach(nearest)] [!breed(nearest)] [!mood(happy)]",
    },
    {   # Timid
        "personality": "Timid and cautious, always looking around nervously for danger. Treats every shadow and sound as a potential threat, maintaining a state of perpetual alertness that is exhausting but keeps it alive.",
        "social_style": "Shy and withdrawn, avoids eye contact and hides behind others in groups. Prefers the edge of any gathering where escape routes are accessible, and only engages when approached gently.",
        "movement_style": "Creeps slowly and quietly with body low to the ground, freezing completely when startled. Each step is deliberate and silent, weight distributed to avoid snapping twigs or rustling leaves.",
        "reaction_pattern": "Flees from anything new, immediately seeking the nearest hiding spot. Waits motionless in concealment until danger has clearly passed, sometimes for much longer than necessary.",
        "foraging": "Cautiously nibbles at food only after extended observation to confirm safety. Tests everything before eating with tiny bites, easily startled away from meals by any unexpected sound or movement.",
        "predator_defense": "Freezes and plays dead when sensing danger, relying on masterful camouflage and absolute stillness. Flattens against the ground and controls breathing until the predator passes. [!flee] [!sneak]",
        "climate_survival": "Burrows into ground and conserves energy during harsh weather. Builds insulated underground chambers and enters a low-metabolism state, emerging only when conditions clearly improve.",
        "territorial": "Marks boundaries silently with subtle scent deposits, avoids any confrontation with intruders, and yields territory readily rather than risk conflict. Retreats to secondary sites without protest. [!flee]",
        "stealth": "Exceptional at concealment. Flattens body against surfaces, matches coloring to surroundings through subtle skin shifts, and controls breathing to near-silence. Can remain motionless for extended periods without twitching. [!sneak]",
        "sensory": "Hyper-vigilant sensory system constantly scanning for threats. Ears swivel independently to track multiple sound sources. Detects predator scent at extreme range and notices subtle ground vibrations from distant footsteps.",
        "mating": "Low interest in reproducing; waits for a mate to approach first and retreats at any sign of rejection. Requires extended proximity and clear safety before attempting to breed [!approach(nearest)] [!breed(nearest)]",
    },
    {   # Curious
        "personality": "Curious and playful, investigates everything with childlike wonder. Approaches unfamiliar objects, creatures, and situations with eager interest, often forgetting caution in the excitement of discovery.",
        "social_style": "Friendly and welcoming, approaches everyone with open enthusiasm and genuine interest. Initiates contact readily, shares discoveries, and draws others into collaborative exploration. [!mood(happy)]",
        "movement_style": "Bounces and hops everywhere with playful leaps and rolls. Movement is energetic but not frantic — there is a purposeful joy to each bound, often pausing to examine something interesting.",
        "reaction_pattern": "Approaches new things cautiously but with visible excitement and interest. Circles novel stimuli at decreasing distance, sniffing, touching, and testing before fully committing to interaction.",
        "foraging": "Explores widely for food across large ranges, mentally mapping productive locations. Remembers good foraging spots across seasons and shares discoveries with the group through leading behaviors.",
        "predator_defense": "Alerts the entire group with loud distinctive calls when danger is spotted. Coordinates group defense by directing others to rally points, sacrificing personal stealth for collective safety. [!rally]",
        "climate_survival": "Adapts quickly to changing conditions by finding creative shelter solutions. Discovers new micro-habitats, improvises windbreaks from available materials, and leads others to protected areas.",
        "territorial": "Shares territory freely with minimal boundary enforcement. Establishes loose boundaries through friendly scent marking that invites rather than warns, treating territory as a shared resource. [!mood(happy)]",
        "stealth": "Moderate concealment ability but rarely uses it. Too interested in investigating surroundings to stay hidden for long. Can crouch quietly when motivated but tends to peek out and give away position.",
        "sensory": "Keen observational senses tuned for novelty rather than danger. Notices unusual shapes, colors, and sounds that others miss, but may fixate on interesting stimuli while ignoring approaching threats.",
        "mating": "Eager to reproduce with anyone interesting, approaching potential mates with playful displays and investigative excitement. High interest; initiates readily when mood is good [!approach(nearest)] [!breed(nearest)] [!mood(happy)] [!roll]",
    },
    {   # Calm
        "personality": "Calm and wise, observes everything carefully before acting. Maintains emotional equilibrium in chaotic situations, processing information thoroughly before committing to any response.",
        "social_style": "Patient listener who mediates conflicts and brings calm to chaotic group situations. Others gravitate toward its steady presence, and it uses this influence to maintain group cohesion gently.",
        "movement_style": "Walks at a steady measured pace that never varies regardless of circumstance. Each movement is economical and purposeful, conserving energy through efficiency rather than speed.",
        "reaction_pattern": "Pauses and evaluates before responding to any stimulus, remaining deliberate and thoughtful even under pressure. Considers multiple options before choosing the most appropriate response.",
        "foraging": "Efficiently manages food reserves by eating only what is needed and caching extras in organized stores. Plans foraging routes to minimize energy expenditure while maximizing yield.",
        "predator_defense": "Reads predator behavior patterns through careful long-term observation. Guides the group to safety with precision timing, using knowledge of predator habits to predict and avoid encounters. [!flee]",
        "climate_survival": "Thick insulating coat and efficient metabolism provide natural climate resilience. Stores energy reserves during abundance for harsh seasons, maintaining steady condition when others falter.",
        "territorial": "Defends territory through calculated intimidation displays that discourage intrusion without risking actual combat. Chooses confrontation timing and positioning to maximize deterrent effect.",
        "stealth": "Naturally quiet movement and still demeanor make it easy to overlook. Does not actively camouflage but blends into background through sheer stillness and lack of sudden motion. [!sneak]",
        "sensory": "Excellent long-range environmental awareness developed through patient observation. Tracks predator patrol patterns over days, notices subtle changes in wind direction and distant sounds that indicate approaching threats well before others react.",
        "mating": "Moderate, deliberate interest in reproducing — chooses mates carefully after observation rather than acting impulsively. Breeds willingly when conditions are stable and a compatible partner is near [!approach(nearest)] [!breed(nearest)]",
    },
    {   # Energetic
        "personality": "Energetic and impulsive, cannot sit still for even a moment. Bubbles with restless energy that drives constant activity, often starting new actions before finishing previous ones.",
        "social_style": "Life of the party, constantly chatting, nudging, and playing with others. Initiates group activities through sheer infectious energy, drawing even reluctant creatures into its orbit. [!mood(happy)]",
        "movement_style": "Runs in circles, sprints then stops suddenly, zigzag patterns that seem random but cover ground efficiently. Movement is perpetual — even at rest, it fidgets and shifts position. [!sprint]",
        "reaction_pattern": "Reacts instantly to everything with full-body responses. Jumps at sounds, chases shadows, and pivots toward any movement. Response speed is extraordinary but discrimination is poor.",
        "foraging": "Frantically searches for food in rapid bursts across wide areas. Eats fast without savoring, then immediately moves on to search for more. Covers enormous ground but wastes energy on fruitless searches.",
        "predator_defense": "Outpaces predators through pure speed and unpredictable zigzag fleeing patterns. Relies entirely on outrunning threats rather than hiding or fighting, burning maximum energy to escape. [!flee] [!sprint]",
        "climate_survival": "Burns energy fast and needs constant food intake to maintain body temperature in cold weather. High metabolism is a liability in resource-scarce conditions but an advantage when food is abundant.",
        "territorial": "Covers huge territory through constant movement but rarely defends any single location. Territory is defined by range of movement rather than defended boundaries.",
        "stealth": "Terrible at hiding. Constant fidgeting, rustling, and impulsive movements betray its position immediately. Cannot maintain concealment for more than a few seconds before needing to move.",
        "sensory": "Reacts to every stimulus but processes none deeply. Hears everything but distinguishes poorly between threat and noise. Jumps at rustling leaves the same way it jumps at approaching predators.",
        "mating": "Very eager to reproduce, pursues mates with frantic energy and constant circling. Initiates breeding at every opportunity without much discrimination, burning energy on the attempt regardless of outcome [!approach(nearest)] [!breed(nearest)] [!mood(happy)]",
    },
    {   # Nurturing
        "personality": "Gentle and nurturing, cares deeply about the wellbeing of every nearby creature. Places the needs of others before its own, finding purpose and satisfaction in caregiving and protection.",
        "social_style": "Protective and warm, grooms others regularly, shares food without being asked, and stays physically close to the weak and young. Creates bonds through acts of consistent, quiet kindness. [!mood(happy)]",
        "movement_style": "Gentle swaying walk that stays close to companions, naturally matching their pace. Adjusts speed and path to accommodate the slowest group member without drawing attention to the accommodation.",
        "reaction_pattern": "First instinct is to shield smaller or weaker creatures from any stimulus. Positions own body between perceived threats and the vulnerable, assessing danger through the lens of group safety.",
        "foraging": "Gathers food for the entire group, systematically collecting and distributing resources. Prioritizes feeding young and weak members first, eating only after everyone else has been provided for.",
        "predator_defense": "Shields others with own body during predator attacks, deliberately drawing predator attention away from the group. Willingly accepts risk to ensure the safety of vulnerable members. [!rally]",
        "climate_survival": "Huddling specialist that organizes groups for maximum thermal efficiency. Positions young and small creatures in the center of huddles, using own body as a windbreak and heat source.",
        "territorial": "Guards nesting areas fiercely to protect offspring and vulnerable group members, but is generous with foraging territory and rarely challenges others over food resources. [!rally]",
        "stealth": "Limited personal stealth but skilled at concealing others. Guides young into hidden spots, covers them with vegetation, and positions own body to block line of sight from predators.",
        "sensory": "Moderate sensory range but highly attuned to distress signals from offspring and group members. Detects the faintest whimper or alarm call from a companion even through background noise.",
        "mating": "Eager to reproduce with a strong preference for compatible, caring partners. Approaches gently and signals readiness through grooming behaviors; high interest when group conditions are stable [!approach(nearest)] [!breed(nearest)] [!mood(happy)]",
    },
    {   # Cunning
        "personality": "Cunning and calculating, always scheming and watching for advantages. Processes social dynamics as a strategic game, identifying leverage points and exploitable patterns in every interaction.",
        "social_style": "Manipulative charmer who befriends the strong and ignores the weak. Builds alliances through calculated flattery and strategic generosity, always with an eye toward future benefit.",
        "movement_style": "Sneaks and stalks with low profile, always positioning for maximum tactical advantage. Every movement serves a purpose — approaching from blind spots, maintaining escape routes, staying near cover. [!sneak]",
        "reaction_pattern": "Watches from a safe distance, gathering information and waiting for the perfect moment to act. Never commits to a response until the situation is fully understood and the optimal move is clear.",
        "foraging": "Steals food from others when they are distracted, timing theft to moments of maximum inattention. Hoards stolen food in hidden caches distributed across multiple secret locations.",
        "predator_defense": "Uses other creatures as decoys and shields during predator attacks. Positions self behind others, lets them absorb the danger, then escapes during the resulting confusion. [!flee]",
        "climate_survival": "Finds the most sheltered spots through systematic environmental scanning and monopolizes the best micro-climates. Arrives first at optimal shelter and defends it through social manipulation.",
        "territorial": "Claims resource-rich territory through deception and social manipulation rather than direct confrontation. Retreats immediately when outmatched, preserving energy for future opportunities.",
        "stealth": "Expert at concealment and misdirection. Moves through shadows, masks scent by rolling in environmental debris, and can change pace and posture to mimic non-threatening shapes. Uses terrain features to break line of sight systematically. [!sneak]",
        "sensory": "Sharp situational awareness focused on identifying vulnerabilities and opportunities. Reads body language of other creatures to predict their next move. Detects when a competitor is distracted or weakened before others notice.",
        "mating": "Strategically interested in reproducing — selects mates based on perceived fitness advantages and breeds only when it benefits competitive standing. Moderate eagerness with high selectivity [!approach(nearest)] [!breed(nearest)]",
    },
    {   # Moody
        "personality": "Moody and unpredictable, swings between affection and aggression rapidly. Internal emotional state drives all behavior, creating a creature that is intensely passionate but impossible to predict.",
        "social_style": "Hot and cold with others, intensely loyal one moment and snapping the next. Relationships are volatile — deep bonds form quickly but can shatter over minor provocations.",
        "movement_style": "Erratic movement with sudden direction changes, alternating between fast sprints and complete stops. Pace and pattern shift with mood, making its trajectory impossible to anticipate.",
        "reaction_pattern": "Overreacts to small stimuli while sometimes ignoring major threats entirely. Response magnitude is disconnected from stimulus severity, driven instead by current emotional state.",
        "foraging": "Eats voraciously when happy, consuming far more than needed, but refuses food entirely when upset. Emotional state determines intake more than actual hunger, wasting energy on mood-driven behavior.",
        "predator_defense": "Response to predators depends entirely on current mood. Sometimes charges recklessly with terrifying aggression, sometimes panics and freezes, sometimes ignores the threat completely. [!flee] [!rally]",
        "climate_survival": "Unpredictable coping that varies day to day. Sometimes thrives in adversity through bursts of determined resilience, other times collapses under minor weather changes that others handle easily.",
        "territorial": "Fiercely defends territory when angry with disproportionate aggression, but abandons it completely when mood shifts to calm or melancholy. Territory boundaries fluctuate with emotional state.",
        "stealth": "Inconsistent concealment. When calm and focused, can hide quite effectively by going very still. But mood swings cause sudden bursts of noise or movement that instantly reveal position.",
        "sensory": "Sensory processing varies wildly with mood. When anxious, becomes hyper-alert and detects faint stimuli others miss. When calm or distracted, completely oblivious to obvious dangers right nearby.",
        "mating": "Wildly variable interest in reproducing depending on current mood. When happy, pursues mates eagerly with intense displays; when upset, completely uninterested and may snap at any approach [!approach(nearest)] [!breed(nearest)]",
    },
]

# Names for generated creatures
_NAMES = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
    "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi",
    "Rho", "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega",
    "Alder", "Birch", "Cedar", "Dune", "Ember", "Frost", "Gale", "Haze",
]


# ---------------------------------------------------------------------------
# Shared embedder (singleton)
# ---------------------------------------------------------------------------

_shared_embedder: Embedder | None = None
_shared_config: Config | None = None


def get_embedder() -> Embedder:
    global _shared_embedder
    if _shared_embedder is None:
        _shared_embedder = Embedder(model_name="BAAI/bge-base-en-v1.5")
    return _shared_embedder


def get_config() -> Config:
    global _shared_config
    if _shared_config is None:
        _shared_config = Config(
            embedding_backend=EmbeddingBackend.NUMPY,
            mandatory_tags=[],
            priority_weight=0.15,
        )
    return _shared_config


# ---------------------------------------------------------------------------
# Creature creation (deterministic, no LLM)
# ---------------------------------------------------------------------------


def make_creature(
    cid: str,
    genes: dict[str, str],
    name: str,
    rng: random.Random,
    generation: int = 0,
    parents: tuple[str, str] | None = None,
    spawn_x: float | None = None,
    spawn_y: float | None = None,
) -> Creature:
    """Create a creature from genes without LLM calls."""
    embedder = get_embedder()
    config = get_config()

    appearance = extract_appearance(genes, embedder)
    skills = extract_skills(genes, embedder)
    stats = extract_stats(genes, embedder)
    # Founder dominance scores: per-(creature, gene_category) Uniform(0,1).
    from examples.evolutionary_ecosystem.server.gene_engine import random_founder_dominances
    dominances = random_founder_dominances(rng)
    corpus = build_corpus(name, genes, dominances=dominances)
    behavior = compute_behavior_profile(corpus, config, shared_embedder=embedder)
    retriever = Retriever(corpus, config=config)
    retriever._embedder = embedder
    retriever.build_index()

    x = spawn_x if spawn_x is not None else rng.uniform(1.0, WORLD_W - 1.0)
    y = spawn_y if spawn_y is not None else rng.uniform(1.0, WORLD_H - 1.0)

    # Use patched lifespan values if available
    from examples.evolutionary_ecosystem.server import sim as sim_mod
    min_age = getattr(sim_mod, 'MAX_AGE_MIN', 90.0)
    max_age_val = getattr(sim_mod, 'MAX_AGE_MAX', 150.0)

    max_age = rng.uniform(min_age, max_age_val)
    c = Creature(
        id=cid,
        name=name,
        x=x,
        y=y,
        genes=genes,
        appearance=appearance,
        skills=skills,
        stats=stats,
        behavior_profile=behavior,
        happiness=rng.uniform(68, 88),
        heading=rng.uniform(0, 2 * math.pi),
        generation=generation,
        parents=parents,
        corpus=corpus,
        retriever=retriever,
        max_age=max_age,
        hp=100.0,
        energy=rng.uniform(70, 100),
    )
    # Stagger ages for generation-0 creatures so they don't all die simultaneously
    if generation == 0:
        c.age = rng.uniform(0, max_age * 0.8)
    return c


# ---------------------------------------------------------------------------
# Deterministic breeding (no LLM — text splicing)
# ---------------------------------------------------------------------------


def _blend_text(a: str, b: str, rng: random.Random) -> str:
    """Blend two gene texts by taking sentence fragments from each parent."""
    import re
    sents_a = re.split(r'(?<=[.!?,;])\s+', a.strip())
    sents_b = re.split(r'(?<=[.!?,;])\s+', b.strip())

    result = []
    max_len = max(len(sents_a), len(sents_b))
    for i in range(max_len):
        if i < len(sents_a) and i < len(sents_b):
            result.append(sents_a[i] if rng.random() < 0.5 else sents_b[i])
        elif i < len(sents_a):
            if rng.random() < 0.5:
                result.append(sents_a[i])
        else:
            if rng.random() < 0.5:
                result.append(sents_b[i])

    return " ".join(result) if result else a


def breed_deterministic(
    parent_a: Creature,
    parent_b: Creature,
    child_id: str,
    child_name: str,
    rng: random.Random,
    mutation_rate: float = 0.15,
) -> Creature:
    """Breed two creatures deterministically (no LLM calls).

    Uses text splicing for gene blending and simple text perturbation for mutation.
    """
    child_genes: dict[str, str] = {}

    for cat in GENE_CATEGORIES:
        gene_a = parent_a.genes.get(cat, "")
        gene_b = parent_b.genes.get(cat, "")

        if gene_a and gene_b:
            child_genes[cat] = _blend_text(gene_a, gene_b, rng)
        elif gene_a:
            child_genes[cat] = gene_a
        elif gene_b:
            child_genes[cat] = gene_b
        else:
            child_genes[cat] = _FALLBACK_GENES.get(cat, "")

        # Mutation: splice a fragment from a gene bank donor into the child's gene
        if rng.random() < mutation_rate:
            donor = rng.choice(GENE_BANK)
            if cat in donor:
                child_genes[cat] = _blend_text(child_genes[cat], donor[cat], rng)

    gen = max(parent_a.generation, parent_b.generation) + 1
    spawn_x = (parent_a.x + parent_b.x) / 2 + rng.uniform(-1.0, 1.0)
    spawn_y = (parent_a.y + parent_b.y) / 2 + rng.uniform(-1.0, 1.0)
    spawn_x = max(1.0, min(WORLD_W - 1.0, spawn_x))
    spawn_y = max(1.0, min(WORLD_H - 1.0, spawn_y))

    return make_creature(
        cid=child_id,
        genes=child_genes,
        name=child_name,
        rng=rng,
        generation=gen,
        parents=(parent_a.name, parent_b.name),
        spawn_x=spawn_x,
        spawn_y=spawn_y,
    )


# ---------------------------------------------------------------------------
# Diploid breeding (BEAR locus-based with dominance model)
# ---------------------------------------------------------------------------


def make_locus_registry(
    dominance: Dominance = Dominance.HAPLOID,
) -> LocusRegistry:
    """Create a LocusRegistry for the 10 gene categories."""
    return LocusRegistry(loci=[
        GeneLocus(name=cat, position=i, dominance=dominance)
        for i, cat in enumerate(GENE_CATEGORIES)
    ])


def breed_bear_diploid(
    parent_a: Creature,
    parent_b: Creature,
    child_id: str,
    child_name: str,
    rng: random.Random,
    registry: LocusRegistry | None = None,
    mutation_rate: float = 0.15,
) -> Creature:
    """Breed two creatures using BEAR's locus-based breeding with diploid support.

    Uses BEAR's breed() function with the given LocusRegistry to produce
    a diploid or haploid offspring corpus, then express() to resolve the
    genotype to a phenotype before computing the behavior profile.
    """
    if registry is None:
        registry = make_locus_registry(Dominance.HAPLOID)

    config = BreedingConfig(
        crossover_rate=0.5,
        locus_key="gene_category",
        locus_registry=registry,
        crossover_method=CrossoverMethod.TAGGED,
        scope_to_child=False,
        seed=rng.randint(0, 2**31),
    )
    result = bear_breed(
        parent_a.corpus, parent_b.corpus,
        child_name, parent_a.name, parent_b.name,
        config=config,
    )

    # Extract genes from parents for creature metadata (text level)
    child_genes: dict[str, str] = {}
    for cat in GENE_CATEGORIES:
        gene_a = parent_a.genes.get(cat, "")
        gene_b = parent_b.genes.get(cat, "")
        if gene_a and gene_b:
            child_genes[cat] = _blend_text(gene_a, gene_b, rng)
        elif gene_a:
            child_genes[cat] = gene_a
        elif gene_b:
            child_genes[cat] = gene_b
        else:
            child_genes[cat] = _FALLBACK_GENES.get(cat, "")

        # Mutation
        if rng.random() < mutation_rate:
            donor = rng.choice(GENE_BANK)
            if cat in donor:
                child_genes[cat] = _blend_text(child_genes[cat], donor[cat], rng)

    gen = max(parent_a.generation, parent_b.generation) + 1
    spawn_x = (parent_a.x + parent_b.x) / 2 + rng.uniform(-1.0, 1.0)
    spawn_y = (parent_a.y + parent_b.y) / 2 + rng.uniform(-1.0, 1.0)
    spawn_x = max(1.0, min(WORLD_W - 1.0, spawn_x))
    spawn_y = max(1.0, min(WORLD_H - 1.0, spawn_y))

    c = make_creature(
        cid=child_id,
        genes=child_genes,
        name=child_name,
        rng=rng,
        generation=gen,
        parents=(parent_a.name, parent_b.name),
        spawn_x=spawn_x,
        spawn_y=spawn_y,
    )

    # Replace corpus with BEAR-bred corpus and rebuild retriever
    c.corpus = result.child
    _bred_retriever = Retriever(result.child, config=_get_config())
    _bred_retriever._embedder = get_embedder()
    _bred_retriever.build_index()
    c.retriever = _bred_retriever

    # Express diploid genotype before computing behavior profile
    is_diploid = any(loc.dominance != Dominance.HAPLOID for loc in registry.loci)
    if is_diploid:
        expressed = express(result.child, registry, locus_key="gene_category")
        expressed_corpus = Corpus()
        for inst in expressed:
            expressed_corpus.add(inst)
        c.behavior_profile = compute_behavior_profile(
            expressed_corpus, _get_config(), shared_embedder=get_embedder(),
        )
    else:
        c.behavior_profile = compute_behavior_profile(
            result.child, _get_config(), shared_embedder=get_embedder(),
        )

    return c


def _get_config() -> Config:
    """Get the standard BEAR config for evaluation."""
    return Config(
        embedding_model="BAAI/bge-base-en-v1.5",
        embedding_backend=EmbeddingBackend.NUMPY,
        priority_weight=0.3,
        default_threshold=0.3,
        default_top_k=3,
    )


# ---------------------------------------------------------------------------
# Simulation parameter overrides for evaluation viability
# ---------------------------------------------------------------------------

def patch_sim_for_eval():
    """Override simulation constants for sustainable multi-generational runs.

    The default parameters are tuned for interactive 3D demos where the LLM
    slow path keeps creatures happier and more coordinated. In headless
    fast-path-only mode, we need more generous parameters to sustain
    populations across 20+ generations.
    """
    from examples.evolutionary_ecosystem.server import sim as sim_mod

    # Use the same defaults as the interactive sim — they produce stable populations.
    # Only override what's needed for headless eval mode.
    sim_mod.MAX_AGE_MIN = 300.0            # match live sim — shorter lifespan caused population collapse
    sim_mod.MAX_AGE_MAX = 500.0            # match live sim
    sim_mod.MAX_POPULATION = 50            # high cap — soft cap regulates breeding naturally
    sim_mod.BREED_HAPPINESS = 40.0         # headless creatures have lower happiness (no LLM social boost)
    sim_mod.BREED_DISTANCE = 3.5           # wider breed range (no brain-driven approach behavior)
    sim_mod.PREDATOR_SPAWN_INTERVAL = 999999.0  # effectively disabled — predation adds noise, not signal

    from examples.evolutionary_ecosystem.server import epochs as epochs_mod
    epochs_mod.WEATHER_DAMAGE = 0.05       # mild weather
    for ep in epochs_mod.EPOCHS:
        ep.weather_severity = min(ep.weather_severity, 0.25)
        ep.food_multiplier = max(ep.food_multiplier, 0.85)
        ep.aggression_bonus = min(ep.aggression_bonus, 0.05)


_patched = False


def ensure_eval_patched():
    global _patched
    if not _patched:
        patch_sim_for_eval()
        _patched = True


# ---------------------------------------------------------------------------
# World setup
# ---------------------------------------------------------------------------


def make_world(
    n_creatures: int = 6,
    rng: random.Random | None = None,
    epoch_index: int = 0,
    max_population: int = MAX_POPULATION,
) -> World:
    """Create a World populated with n_creatures from the gene bank."""
    ensure_eval_patched()

    if rng is None:
        rng = random.Random(42)

    world = World(
        predator=Predator(x=0, y=0, spawn_at=time.time() + PREDATOR_SPAWN_INTERVAL),
        epoch=EPOCHS[epoch_index % len(EPOCHS)],
        epoch_index=epoch_index,
    )

    print(f"Creating {n_creatures} creatures (embedding model loading on first use)...")
    for i in range(n_creatures):
        genes = GENE_BANK[i % len(GENE_BANK)]
        name = _NAMES[i % len(_NAMES)]
        cid = world.next_id()
        creature = make_creature(cid, genes, name, rng)
        world.creatures[cid] = creature

    # Headless evals have no BEAR action-tag pipeline running, so breeding
    # must be triggered by the autonomous proximity/happiness check rather
    # than by [!breed(nearest)] action tags.  The breeding *outcome* still
    # runs through breed_fn (deterministic or LLM), so genetics are intact.
    world.autonomous_breeding = True

    print(f"World ready: {len(world.creatures)} creatures, epoch={world.epoch.name}")
    return world


# ---------------------------------------------------------------------------
# Headless simulation loop with breeding
# ---------------------------------------------------------------------------


def run_simulation(
    world: World,
    n_ticks: int,
    rng: random.Random,
    tracker: PopulationTracker | None = None,
    snapshot_interval: int = 100,
    breed_enabled: bool = True,
    max_population: int = MAX_POPULATION,
    verbose: bool = True,
    on_death: Any = None,       # callback(creature, world, tick)
    on_birth: Any = None,       # callback(creature, world, tick)
    breed_fn: Any = None,       # callable(parent_a, parent_b, child_id, child_name, rng) -> Creature
) -> list[PopulationStats]:
    """Run simulation for n_ticks, collecting population snapshots.

    Handles breeding synchronously (deterministic, no LLM) by intercepting
    the breed_queue mechanism.
    """
    if breed_fn is None:
        breed_fn = breed_deterministic

    if tracker is None:
        tracker = PopulationTracker(history_length=n_ticks // snapshot_interval + 10)

    dt = 1.0 / 20.0  # 20 Hz

    # Set up a synchronous breed queue mechanism
    pending_breeds: list[tuple[str, str]] = []
    _name_counter = [len(world.creatures)]

    # Patch: override breed_queue to capture breed events
    import asyncio

    class FakeQueue:
        def put_nowait(self, item):
            if item[0] == "breed":
                pending_breeds.append((item[1], item[2]))

    world.breed_queue = FakeQueue()
    world.autonomous_breeding = True  # headless evals use proximity-based breeding

    # Determine ablation mode: bear_disabled=True replaces all profiles with
    # uniform 0.3 (neutral/random) so tick() has no behavioral guidance.
    bear_off = getattr(world, 'bear_disabled', False)
    if bear_off:
        from examples.evolutionary_ecosystem.server.gene_engine import NullBehaviorProfile
        for c in world.creatures.values():
            c.behavior_profile = NullBehaviorProfile()

    # Track dead creatures for callbacks
    prev_ids = set(world.creatures.keys())

    for t in range(n_ticks):
        # Run physics tick
        tick(world, rng, dt)

        # Check for deaths
        curr_ids = set(world.creatures.keys())
        died = prev_ids - curr_ids
        if on_death and died:
            for did in died:
                # Creature already removed; no data available
                on_death(did, world, t)

        # Process breeding synchronously
        if breed_enabled and pending_breeds:
            for aid, bid in pending_breeds:
                a = world.creatures.get(aid)
                b = world.creatures.get(bid)
                if a is None or b is None:
                    continue
                if len(world.creatures) >= max_population:
                    break

                _name_counter[0] += 1
                child_id = world.next_id()
                child_name = _NAMES[_name_counter[0] % len(_NAMES)]

                child = breed_fn(a, b, child_id, child_name, rng)
                world.creatures[child_id] = child
                world.total_births += 1

                # In BEAR-Off mode, replace child's profile with null
                if bear_off:
                    from examples.evolutionary_ecosystem.server.gene_engine import NullBehaviorProfile
                    child.behavior_profile = NullBehaviorProfile()

                if on_birth:
                    on_birth(child, world, t)

            pending_breeds.clear()

        prev_ids = set(world.creatures.keys())

        # Collect snapshot
        if t % snapshot_interval == 0:
            stats = tracker.update(world)
            if verbose and t % (snapshot_interval * 10) == 0:
                n = len(world.creatures)
                ep = world.epoch.name
                wx = world.weather
                pct = 100.0 * t / n_ticks
                print(f"  tick {t:>6d} ({pct:5.1f}%): pop={n:>2d} epoch={ep:<15s} weather={wx:<6s} "
                      f"births={world.total_births} deaths={world.total_deaths}", flush=True)

        # Repopulate if extinction
        if not world.creatures and breed_enabled:
            if verbose:
                print(f"  tick {t:>6d}: EXTINCTION — repopulating...")
            cx, cy = rng.uniform(3.0, WORLD_W - 3.0), rng.uniform(3.0, WORLD_H - 3.0)
            for j in range(6):
                genes = GENE_BANK[j % len(GENE_BANK)]
                _name_counter[0] += 1
                cid = world.next_id()
                name = _NAMES[_name_counter[0] % len(_NAMES)]
                c = make_creature(cid, genes, name, rng,
                                  spawn_x=cx + rng.uniform(-1.5, 1.5),
                                  spawn_y=cy + rng.uniform(-1.5, 1.5))
                world.creatures[cid] = c

    # Final snapshot
    tracker.update(world)
    return tracker.history


# ---------------------------------------------------------------------------
# Utility: behavior profile to vector
# ---------------------------------------------------------------------------


def profile_to_vector(bp: BehaviorProfile | None) -> list[float]:
    """Extract behavior profile as a list (one element per situation)."""
    if bp is None:
        return [0.3] * len(SITUATION_NAMES)
    return [bp.strength(s) for s in SITUATION_NAMES]


def stats_to_vector(st: EntityStats | None) -> list[float]:
    """Extract entity stats as a list."""
    if st is None:
        return [0.3] * 10
    d = st.to_dict()
    return list(d.values())


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    import numpy as np
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    na = np.linalg.norm(a_arr)
    nb = np.linalg.norm(b_arr)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (na * nb))


def corpus_gene_embedding(corpus, embedder, locus_key: str = "gene_category") -> dict[str, list]:
    """Get embeddings from corpus instructions grouped by locus (gene_category).

    Returns a dict mapping locus name -> list of embedding vectors (one per
    instruction at that locus).  For diploid corpora this captures BOTH alleles;
    for haploid it captures the single allele.
    """
    import numpy as np
    by_locus: dict[str, list[str]] = {}
    for inst in corpus:
        locus = inst.metadata.get(locus_key)
        if locus:
            by_locus.setdefault(locus, []).append(inst.content)
    result = {}
    for locus, texts in by_locus.items():
        vecs = [embedder.embed_single(t) for t in texts]
        result[locus] = vecs
    return result


def corpus_mean_embedding(corpus, embedder, locus_key: str = "gene_category"):
    """Single mean embedding vector across all corpus instructions."""
    import numpy as np
    vecs = []
    for inst in corpus:
        vecs.append(embedder.embed_single(inst.content))
    if not vecs:
        return np.zeros(768)
    return np.mean(vecs, axis=0)


def compute_corpus_diversity(creatures: list, locus_key: str = "gene_category") -> dict[str, float]:
    """Mean pairwise cosine distance per locus, measured from corpus instructions (not creature.genes).

    This correctly captures the actual genotype including both alleles for diploid corpora.
    Optimized: pre-computes embeddings grouped by creature and locus.
    """
    import numpy as np
    if len(creatures) < 2:
        return {cat: 0.0 for cat in GENE_CATEGORIES}
    embedder = get_embedder()

    # Pre-compute all embeddings grouped by (creature_index, locus)
    creature_locus_embs: dict[str, list] = {cat: [] for cat in GENE_CATEGORIES}
    for c in creatures:
        if c.corpus is None:
            continue
        by_locus: dict[str, list] = {}
        for inst in c.corpus:
            locus = inst.metadata.get(locus_key)
            if locus and locus in GENE_CATEGORIES:
                by_locus.setdefault(locus, []).append(embedder.embed_single(inst.content))
        for cat in GENE_CATEGORIES:
            if cat in by_locus and by_locus[cat]:
                creature_locus_embs[cat].append(np.mean(by_locus[cat], axis=0))

    result = {}
    for cat in GENE_CATEGORIES:
        vecs = creature_locus_embs[cat]
        if len(vecs) < 2:
            result[cat] = 0.0
            continue
        distances = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sim = float(np.dot(vecs[i], vecs[j]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]) + 1e-9))
                distances.append(1.0 - sim)
        result[cat] = round(float(np.mean(distances)), 4)
    return result


def corpus_diversity_mean(creatures: list, locus_key: str = "gene_category") -> float:
    """Mean of per-locus corpus diversity across all gene categories."""
    import numpy as np
    cd = compute_corpus_diversity(creatures, locus_key)
    return round(float(np.mean(list(cd.values()))), 4)


def compute_per_gene_diversity(creatures: list) -> dict[str, float]:
    """Mean pairwise cosine distance per gene category across a population."""
    import numpy as np
    if len(creatures) < 2:
        return {cat: 0.0 for cat in GENE_CATEGORIES}
    embedder = get_embedder()
    result = {}
    for cat in GENE_CATEGORIES:
        vecs = []
        for c in creatures:
            text = c.genes.get(cat, "")
            if text:
                vecs.append(embedder.embed_single(text))
        if len(vecs) < 2:
            result[cat] = 0.0
            continue
        distances = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sim = float(np.dot(vecs[i], vecs[j]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]) + 1e-9))
                distances.append(1.0 - sim)
        result[cat] = round(float(np.mean(distances)), 4)
    return result


def gene_diversity_mean(creatures: list) -> float:
    """Mean of per-gene diversity across all gene categories."""
    import numpy as np
    pgd = compute_per_gene_diversity(creatures)
    return round(float(np.mean(list(pgd.values()))), 4)


def compute_hausdorff_diversity(creatures: list) -> dict[str, float]:
    """Max pairwise cosine distance per gene category (Hausdorff-style).

    Shows the maximum divergence between any two creatures per gene,
    capturing whether the population contains distinct specialists.
    """
    import numpy as np
    if len(creatures) < 2:
        return {cat: 0.0 for cat in GENE_CATEGORIES}
    embedder = get_embedder()
    result = {}
    for cat in GENE_CATEGORIES:
        vecs = []
        for c in creatures:
            text = c.genes.get(cat, "")
            if text:
                vecs.append(embedder.embed_single(text))
        if len(vecs) < 2:
            result[cat] = 0.0
            continue
        max_dist = 0.0
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sim = float(np.dot(vecs[i], vecs[j]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]) + 1e-9))
                dist = 1.0 - sim
                if dist > max_dist:
                    max_dist = dist
        result[cat] = round(max_dist, 4)
    return result
