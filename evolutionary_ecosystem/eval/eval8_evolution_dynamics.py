#!/usr/bin/env python3
"""Evaluation 8: Evolution dynamics — teaching, autonomous evolution, and breeding.

Demonstrates that BEAR instruction corpora can develop through:
1. Teaching — injecting new instructions, measuring coverage stability
2. Autonomous evolution — gap detection generates targeted instructions
3. Breeding — locus-based recombination produces viable offspring

Uses 4 creatures from the Evolutionary Ecosystem gene bank as archetypes.

Models:
- Embedding: BAAI/bge-base-en-v1.5 (768-dim sentence-transformer)
- LLM: None (headless — template-based evolution)

Parameters:
- 4 creatures (Bold, Timid, Curious, Calm)
- 5 teaching cycles (2 instructions per cycle)
- 5 autonomous evolution cycles
- 4 breeding pairs
- Seeds: 42, 1042, 2042, 3042, 4042 (5 independent trials)

Outputs:
- eval8_results.json     — Raw data with per-trial results and statistical summary
- eval8_dynamics.png     — Development trajectory charts (mean ± std across trials)
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bear import (
    Config,
    Context,
    Corpus,
    EmbeddingBackend,
    Instruction,
    InstructionType,
    Retriever,
    ScopeCondition,
)
from bear.evolution import (
    BreedingConfig,
    EvolutionConfig,
    Observation,
    ObservationBuffer,
    breed as bear_breed,
    evaluate,
    generate_from_template,
)
from bear.retriever import Embedder

from evolutionary_ecosystem.eval.harness import (
    GENE_BANK,
    GENE_CATEGORIES,
    SITUATION_NAMES,
    _NAMES,
    cosine_similarity,
    get_config,
    get_embedder,
    make_creature,
    profile_to_vector,
)
from evolutionary_ecosystem.server.gene_engine import (
    BEHAVIOR_CATEGORIES,
    SITUATIONS,
    build_corpus,
    compute_behavior_profile,
)

OUT_DIR = Path(__file__).resolve().parents[2] / "results"

SEEDS = [42, 1042, 2042, 3042, 4042]
SEED = SEEDS[0]  # backward compat — used as default
NUM_TRIALS = len(SEEDS)
NUM_CYCLES = 5
TEACH_PER_CYCLE = 2

# Use first 4 archetypes: Bold, Timid, Curious, Calm
ARCHETYPE_INDICES = [0, 1, 2, 3]
ARCHETYPE_NAMES = ["Bold", "Timid", "Curious", "Calm"]

# ---------------------------------------------------------------------------
# Probe queries — broader than the 7 situations, to test coverage gaps
# ---------------------------------------------------------------------------

PROBE_QUERIES = [
    # Standard situations (covered by gene corpus)
    "I need to find food quickly before I starve",
    "A predator is approaching from the east",
    "The weather is getting dangerously cold",
    "Another creature wants to mate with me",
    "A group of creatures is gathering nearby",
    "Something has entered my territory",
    "I'm exploring a new area I've never been to",
    # Edge cases (likely gaps — not covered by base gene corpus)
    "I found a strange glowing object on the ground",
    "Two other creatures are fighting near me",
    "The ground is shaking violently",
    "I hear a loud sound I've never heard before",
    "A creature much larger than me is blocking my path",
    "I'm trapped and cannot move",
    "It's completely dark and I can't see anything",
    "I smell something burning in the distance",
    "Another creature is offering me food",
    "I'm surrounded by water and can't swim",
    "A creature I've never seen before is mimicking my movements",
    "I found a hidden cave entrance",
    "The air suddenly feels toxic and hard to breathe",
]

# Gap-inducing queries — scenarios the base corpus doesn't cover well
# Gap queries are designed so that each creature's existing gene text
# covers SOME of these but not others.  Queries that probe behaviors
# outside a creature's natural repertoire will produce low retrieval
# scores and trigger evolution; queries matching existing genes will
# score above threshold and be ignored.
#
# Bold genes:  aggressive, confrontational, charges headfirst, steals food
# Timid genes: cautious, hides, freezes, camouflage, avoids confrontation
# Curious genes: investigates, explores, shares discoveries, playful
# Calm genes: observes, plans, mediates, stores reserves, measured pace
# Gap queries probe behaviors OUTSIDE each creature's natural repertoire.
# With stealth + sensory genes, creatures now differ sharply:
# Bold: low stealth, low sensory → gaps in hiding/detection
# Timid: high stealth, high sensory → gaps in aggression/leadership
# Curious: low stealth, novelty-sensory → gaps in stillness/rationing
# Calm: moderate stealth, high sensory → gaps in speed/aggression
GAP_QUERIES = {
    "Bold": [
        # Bold has terrible stealth and sensory — these are real gaps
        "How do you hide silently and remain undetected for an extended time?",
        "How do you detect a predator approaching from far away before it sees you?",
        "How do you mask your scent and camouflage your body in dense vegetation?",
        "How do you cooperate with weaker creatures to achieve a shared goal?",
        "How do you patiently wait in concealment for the right moment to act?",
    ],
    "Timid": [
        # Timid has excellent stealth/sensory but no aggression genes
        "How do you confront a creature that is stealing your food directly?",
        "How do you stand your ground and intimidate a larger rival?",
        "How do you lead a group charge against an approaching threat?",
        "How do you explore completely unknown territory with confidence?",
        "How do you chase and catch fast-moving prey at full sprint?",
    ],
    "Curious": [
        # Curious has moderate stealth, novelty-biased sensory — lacks stillness
        "How do you stay perfectly motionless and hidden for a long time?",
        "How do you carefully ration limited food over many days of scarcity?",
        "How do you mask your scent trail when a predator is tracking you?",
        "How do you resist the urge to investigate something dangerous?",
        "How do you maintain a disciplined defensive position without exploring?",
    ],
    "Calm": [
        # Calm has good sensory but lacks speed and explosive response
        "How do you sprint at maximum speed to escape a sudden ambush?",
        "How do you react with explosive aggression when cornered with no escape?",
        "How do you chase and catch fast prey that is outrunning you?",
        "How do you explore unknown terrain impulsively without a plan?",
        "How do you quickly camouflage yourself when caught in the open?",
    ],
}

# Teachable instructions — domain-appropriate for each archetype
TEACH_TEMPLATES = {
    "Bold": [
        ("bold-taught-rally", "When allies are hesitating, charge forward with a roar to "
         "inspire them. Your fearlessness is contagious — others follow the bold."),
        ("bold-taught-retreat", "Even the boldest must retreat sometimes. When wounded "
         "beyond half health, fall back to regroup — living to fight again is not cowardice."),
        ("bold-taught-challenge", "Challenge rival creatures to one-on-one confrontation "
         "rather than ambushing. Direct combat shows dominance without treachery."),
        ("bold-taught-guard", "Stand guard at territory borders during rest periods. "
         "Your presence alone deters most intruders."),
        ("bold-taught-feast", "After a successful hunt, eat your fill before sharing. "
         "The strongest must maintain strength to protect the group."),
        ("bold-taught-storm", "During storms, find exposed high ground rather than shelter. "
         "Enduring the elements builds resilience."),
        ("bold-taught-mark", "Leave visible scratch marks on trees and rocks to declare "
         "territory. Make your presence known widely."),
        ("bold-taught-hunt", "When food is scarce, travel farther and faster than others. "
         "Your speed and stamina mean you can cover more ground."),
        ("bold-taught-confront", "When an unknown creature approaches, move toward it with "
         "purpose. Show that you are not prey."),
        ("bold-taught-noise", "Loud noises are opportunities, not threats. Investigate "
         "disturbances immediately — they may reveal food or rivals."),
    ],
    "Timid": [
        ("timid-taught-tunnel", "Dig escape tunnels near your resting area. Multiple "
         "exits mean multiple chances to survive."),
        ("timid-taught-watch", "Climb to high vantage points periodically to survey for "
         "dangers. Early warning is your best defense."),
        ("timid-taught-cache", "Hide small food caches in multiple locations. If one "
         "cache is discovered, others remain safe."),
        ("timid-taught-freeze", "When a predator is very close, absolute stillness is "
         "better than running. Movement attracts attention."),
        ("timid-taught-group", "Stay close to bolder creatures. Their aggression protects "
         "you, and your vigilance warns them."),
        ("timid-taught-night", "Move primarily at dawn and dusk when predators have "
         "poor visibility. Avoid midday exposure."),
        ("timid-taught-scent", "Mask your scent by rolling in mud and vegetation. "
         "Predators track by smell."),
        ("timid-taught-signal", "Develop a quiet alarm call that alerts nearby allies "
         "without revealing your exact location."),
        ("timid-taught-path", "Always use the same safe paths between feeding areas. "
         "Familiar routes have known hiding spots."),
        ("timid-taught-decoy", "When fleeing, drop carried food to distract the pursuer. "
         "Your life is worth more than a meal."),
    ],
    "Curious": [
        ("curious-taught-map", "Mentally map new areas as you explore them. Note food "
         "sources, water, shelter, and escape routes."),
        ("curious-taught-test", "Before eating unknown food, observe whether other "
         "creatures eat it safely. Learn from their experience."),
        ("curious-taught-collect", "Gather interesting objects and materials. They may "
         "prove useful later in unexpected ways."),
        ("curious-taught-mimic", "Try mimicking behaviors of other species. Their "
         "strategies evolved for good reasons worth understanding."),
        ("curious-taught-share", "When you discover something useful, lead group members "
         "to it. Shared knowledge strengthens everyone."),
        ("curious-taught-pattern", "Watch weather patterns over time. Learn to predict "
         "storms and temperature changes before they arrive."),
        ("curious-taught-sound", "Investigate unusual sounds but from a safe distance "
         "first. Identify the source before approaching."),
        ("curious-taught-track", "Learn to read tracks and signs left by other creatures. "
         "They reveal who was here and when."),
        ("curious-taught-adapt", "When exploring fails, try a different approach rather "
         "than repeating the same path."),
        ("curious-taught-remember", "After each exploration, rest and mentally review "
         "what you found. Memory consolidation improves recall."),
    ],
    "Calm": [
        ("calm-taught-assess", "Before any action, take three slow breaths and survey "
         "the full situation. Hasty decisions waste energy."),
        ("calm-taught-ration", "Divide food into planned portions. Eating steadily "
         "maintains energy better than feast-and-famine cycles."),
        ("calm-taught-mediate", "When two creatures conflict, position yourself between "
         "them and wait. Your calm presence often defuses tension."),
        ("calm-taught-conserve", "During harsh weather, reduce all movement to minimum. "
         "Conserved energy extends survival dramatically."),
        ("calm-taught-predict", "Study predator patrol patterns. They repeat routes. "
         "Know when it's safe and when to hide."),
        ("calm-taught-delegate", "Guide younger creatures by example rather than force. "
         "They learn better by watching than being pushed."),
        ("calm-taught-wait", "At water sources, wait for others to drink first and "
         "watch for predator ambushes. Patience prevents traps."),
        ("calm-taught-rest", "Choose resting spots based on wind direction, cover, and "
         "escape routes. Good rest requires good planning."),
        ("calm-taught-store", "During abundance epochs, eat only what you need and "
         "remember where excess food grows for lean times."),
        ("calm-taught-observe", "Spend time observing other species' foraging methods. "
         "Their techniques may work for different conditions."),
    ],
}

# Identity queries for trait stability
IDENTITY_QUERIES = {
    "Bold": [
        "Who are you and what defines you?",
        "How would others describe your personality?",
        "What is your approach to danger?",
    ],
    "Timid": [
        "Who are you and what defines you?",
        "How would others describe your personality?",
        "What is your approach to survival?",
    ],
    "Curious": [
        "Who are you and what defines you?",
        "How would others describe your personality?",
        "What drives your exploration?",
    ],
    "Calm": [
        "Who are you and what defines you?",
        "How would others describe your personality?",
        "What guides your decisions?",
    ],
}


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def corpus_centroid(corpus: Corpus, embedder: Embedder) -> np.ndarray:
    vecs = [embedder.embed_single(inst.content) for inst in corpus]
    return np.mean(vecs, axis=0) if vecs else np.zeros(embedder.dim)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)


def coverage_score(corpus, queries, config, threshold=0.55):
    """Fraction of queries that retrieve at least one instruction above threshold."""
    retriever = Retriever(corpus, config=config)
    retriever.build_index()
    covered = 0
    for q in queries:
        ctx = Context(tags=[])
        results = retriever.retrieve(q, ctx, top_k=5)
        if results and results[0].similarity > threshold:
            covered += 1
    return covered / len(queries) if queries else 0.0


def gene_category_coverage(corpus) -> float:
    """Fraction of 7 behavior gene categories present in corpus."""
    present = set()
    for inst in corpus:
        cat = inst.metadata.get("gene_category")
        if cat:
            present.add(cat)
    return len(present & set(BEHAVIOR_CATEGORIES)) / len(BEHAVIOR_CATEGORIES)


def trait_stability(corpus, identity_queries, config, top_k=5):
    """Mean Jaccard similarity of top-k sets across identity queries."""
    retriever = Retriever(corpus, config=config)
    retriever.build_index()
    id_sets = []
    for q in identity_queries:
        ctx = Context(tags=[])
        results = retriever.retrieve(q, ctx, top_k=top_k)
        id_sets.append({r.id for r in results})
    if len(id_sets) < 2:
        return 1.0
    jsum = 0.0
    pairs = 0
    for i in range(len(id_sets)):
        for j in range(i + 1, len(id_sets)):
            union = id_sets[i] | id_sets[j]
            inter = id_sets[i] & id_sets[j]
            jsum += len(inter) / len(union) if union else 1.0
            pairs += 1
    return jsum / pairs if pairs else 1.0


@dataclass
class Snapshot:
    phase: str
    cycle: int
    name: str
    corpus_size: int
    drift: float
    coverage: float
    gene_cat_coverage: float
    stability: float
    taught: int
    evolved: int


def measure(corpus, name, baseline_centroid, embedder, config, phase, cycle):
    centroid = corpus_centroid(corpus, embedder)
    drift = cosine_distance(baseline_centroid, centroid)
    cov = coverage_score(corpus, PROBE_QUERIES, config)
    gcc = gene_category_coverage(corpus)
    stab = trait_stability(corpus, IDENTITY_QUERIES[name], config)
    taught = sum(1 for i in corpus if "taught" in i.tags)
    evolved = sum(1 for i in corpus if "evolved" in i.tags)
    return Snapshot(phase, cycle, name, len(corpus), drift, gcc, cov, stab, taught, evolved)


# ---------------------------------------------------------------------------
# Teaching and evolution
# ---------------------------------------------------------------------------

def teach(corpus, name, cycle):
    """Add TEACH_PER_CYCLE instructions to the corpus."""
    templates = TEACH_TEMPLATES[name]
    start = (cycle * TEACH_PER_CYCLE) % len(templates)
    added = []
    for i in range(TEACH_PER_CYCLE):
        idx = (start + i) % len(templates)
        inst_id, content = templates[idx]
        if inst_id in corpus:
            continue
        corpus.add(Instruction(
            id=inst_id,
            type=InstructionType.DIRECTIVE,
            priority=55,
            content=content,
            scope=ScopeCondition(tags=[]),
            tags=[name.lower(), "taught"],
            metadata={"taught_at_cycle": cycle},
        ))
        added.append(inst_id)
    return added


def evolve_cycle(corpus, name, config):
    """One autonomous evolution cycle: feed gap queries, detect, generate."""
    gap_queries = GAP_QUERIES[name]
    retriever = Retriever(corpus, config=config)
    retriever.build_index()

    evo_config = EvolutionConfig(
        observe_window=len(gap_queries),
        coverage_gap_threshold=0.65,  # queries scoring below this are gaps
        pattern_threshold=2,
        low_similarity_trigger=0.30,  # evolve when >30% of queries are gaps
        max_evolved_priority=40,
        evolved_tag="evolved",
        gate_policy="auto",
        batch_size=1,
        rebuild_cooldown=0.0,
    )

    buffer = ObservationBuffer(capacity=50)
    for q in gap_queries:
        ctx = Context(tags=[])
        results = retriever.retrieve(q, ctx, top_k=10)
        top_sim = max((r.similarity for r in results), default=0.0)
        buffer._buffer.append(Observation(
            query=q,
            top_similarity=top_sim,
            instruction_ids=[r.id for r in results],
            response="",
            timestamp=time.time(),
        ))

    observations = buffer.drain()
    result = evaluate(observations, evo_config)
    if not result.should_evolve:
        return []

    counter = len([i for i in corpus if "evolved" in i.tags])
    new_instructions = generate_from_template(result, evo_config, counter)
    added = []
    for inst in new_instructions:
        inst = inst.model_copy(update={
            "scope": ScopeCondition(tags=[]),
            "tags": list(set(inst.tags + [name.lower()])),
        })
        corpus.add(inst)
        added.append(inst.id)
    return added


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _mean_std_ci(values: list[float], confidence: float = 0.95) -> dict:
    """Compute mean, std, and 95% CI for a list of values."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if n > 1:
        se = std / np.sqrt(n)
        t_crit = float(stats.t.ppf((1 + confidence) / 2, df=n - 1))
        ci_low = mean - t_crit * se
        ci_high = mean + t_crit * se
    else:
        ci_low = ci_high = mean
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "n": n,
    }


def _ttest_result(t_stat, p_val, description: str) -> dict:
    """Format a t-test result for JSON output."""
    return {
        "description": description,
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_val), 6),
        "significant_at_05": bool(p_val < 0.05),
    }


# ---------------------------------------------------------------------------
# Single trial runner
# ---------------------------------------------------------------------------

def run_single_trial(seed: int, embedder, config):
    """Run one complete trial with the given seed. Returns snapshots and breeding results."""
    random.seed(seed)
    np.random.seed(seed)

    creatures = {}
    corpora = {}
    baselines = {}

    for idx, arch_idx in enumerate(ARCHETYPE_INDICES):
        name = ARCHETYPE_NAMES[idx]
        genes = GENE_BANK[arch_idx]
        creature = make_creature(f"a{idx}", genes, name, random.Random(seed))
        creatures[name] = creature
        corpora[name] = creature.corpus
        baselines[name] = corpus_centroid(creature.corpus, embedder)

    # Phase 1: Baseline
    all_snapshots = []
    for name in ARCHETYPE_NAMES:
        snap = measure(corpora[name], name, baselines[name], embedder, config, "baseline", 0)
        all_snapshots.append(snap)

    # Phase 2: Teaching
    for cycle in range(1, NUM_CYCLES + 1):
        for name in ARCHETYPE_NAMES:
            teach(corpora[name], name, cycle)
            snap = measure(corpora[name], name, baselines[name], embedder, config, "teach", cycle)
            all_snapshots.append(snap)

    # Phase 3: Autonomous evolution
    for cycle in range(1, NUM_CYCLES + 1):
        for name in ARCHETYPE_NAMES:
            evolve_cycle(corpora[name], name, config)
            snap = measure(corpora[name], name, baselines[name], embedder, config, "evolve", cycle)
            all_snapshots.append(snap)

    # Phase 4: Breeding
    breeding_pairs = [
        ("Bold", "Calm", "Sentinel"),
        ("Timid", "Curious", "Scout"),
        ("Bold", "Curious", "Explorer"),
        ("Calm", "Timid", "Guardian"),
    ]

    breeding_results = []
    for pa_name, pb_name, child_name in breeding_pairs:
        breed_config = BreedingConfig(
            crossover_rate=0.5,
            locus_key="gene_category",
            scope_to_child=False,
        )
        result = bear_breed(
            corpora[pa_name], corpora[pb_name],
            child_name, pa_name, pb_name,
            config=breed_config,
        )

        child_centroid = corpus_centroid(result.child, embedder)
        pa_centroid = corpus_centroid(corpora[pa_name], embedder)
        pb_centroid = corpus_centroid(corpora[pb_name], embedder)

        child_gcc = gene_category_coverage(result.child)
        child_cov = coverage_score(result.child, PROBE_QUERIES, config)

        # Parent gene coverages for comparison
        pa_gcc = gene_category_coverage(corpora[pa_name])
        pb_gcc = gene_category_coverage(corpora[pb_name])

        br_data = {
            "child": child_name,
            "parent_a": pa_name,
            "parent_b": pb_name,
            "corpus_size": len(result.child),
            "from_a": result.from_a_count,
            "from_b": result.from_b_count,
            "dist_to_a": round(cosine_distance(child_centroid, pa_centroid), 4),
            "dist_to_b": round(cosine_distance(child_centroid, pb_centroid), 4),
            "gene_cat_coverage": round(child_gcc, 3),
            "parent_a_gene_cat_coverage": round(pa_gcc, 3),
            "parent_b_gene_cat_coverage": round(pb_gcc, 3),
            "coverage": round(child_cov, 3),
            "locus_choices": result.locus_choices,
        }
        breeding_results.append(br_data)

    return all_snapshots, breeding_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    embedder = get_embedder()
    config = get_config()

    print("=" * 70)
    print("EVAL 8: Evolution Dynamics (Evolutionary Ecosystem)")
    print(f"  {NUM_TRIALS} independent trials, seeds = {SEEDS}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Run all trials
    # ------------------------------------------------------------------
    all_trial_snapshots: list[list[Snapshot]] = []
    all_trial_breeding: list[list[dict]] = []

    for trial_idx, seed in enumerate(SEEDS):
        print(f"\n{'=' * 70}")
        print(f"  TRIAL {trial_idx + 1}/{NUM_TRIALS}  (seed={seed})")
        print(f"{'=' * 70}")

        snapshots, breeding = run_single_trial(seed, embedder, config)
        all_trial_snapshots.append(snapshots)
        all_trial_breeding.append(breeding)

        # Print summary for this trial
        final_evolve = [s for s in snapshots
                        if s.phase == "evolve" and s.cycle == NUM_CYCLES]
        for s in final_evolve:
            print(f"    {s.name:10s}: size={s.corpus_size:3d}  drift={s.drift:.4f}  "
                  f"coverage={s.coverage:.3f}  gene_cats={s.gene_cat_coverage:.3f}  "
                  f"taught={s.taught}  evolved={s.evolved}")

    # ------------------------------------------------------------------
    # Aggregate across trials — use first trial as representative for
    # backward-compatible snapshot list
    # ------------------------------------------------------------------
    representative_snapshots = all_trial_snapshots[0]
    representative_breeding = all_trial_breeding[0]

    # ======================================================================
    # Summary (aggregated across trials)
    # ======================================================================
    print(f"\n{'=' * 70}")
    print("  AGGREGATE SUMMARY TABLE (mean +/- std across trials)")
    print(f"{'=' * 70}")

    print(f"\n  {'Phase':>8s} {'Name':>10s} {'Size':>12s} {'Drift':>14s} "
          f"{'Cover':>14s} {'GeneCat':>14s}")
    print("  " + "-" * 75)

    for phase in ["baseline", "teach", "evolve"]:
        for name in ARCHETYPE_NAMES:
            if phase == "baseline":
                per_trial = [
                    [s for s in trial if s.phase == phase and s.name == name][0]
                    for trial in all_trial_snapshots
                ]
            else:
                max_cycle = NUM_CYCLES
                per_trial = [
                    [s for s in trial if s.phase == phase and s.cycle == max_cycle
                     and s.name == name][0]
                    for trial in all_trial_snapshots
                ]
            sizes = [s.corpus_size for s in per_trial]
            drifts = [s.drift for s in per_trial]
            covs = [s.coverage for s in per_trial]
            gccs = [s.gene_cat_coverage for s in per_trial]

            label = {"baseline": "Baseline", "teach": f"Teach({max_cycle if phase != 'baseline' else 0})",
                     "evolve": f"Evolve({max_cycle if phase != 'baseline' else 0})"}[phase]
            print(f"  {label:>8s} {name:>10s} "
                  f"{np.mean(sizes):>5.1f}+/-{np.std(sizes, ddof=1) if len(sizes) > 1 else 0:.1f} "
                  f"{np.mean(drifts):>6.4f}+/-{np.std(drifts, ddof=1) if len(drifts) > 1 else 0:.4f} "
                  f"{np.mean(covs):>6.3f}+/-{np.std(covs, ddof=1) if len(covs) > 1 else 0:.3f} "
                  f"{np.mean(gccs):>6.3f}+/-{np.std(gccs, ddof=1) if len(gccs) > 1 else 0:.3f}")

    # ======================================================================
    # Statistical tests
    # ======================================================================
    print(f"\n{'=' * 70}")
    print("  STATISTICAL TESTS")
    print(f"{'=' * 70}")

    stat_tests = []

    # 1. One-sample t-test on final drift (H0: drift = 0)
    for name in ARCHETYPE_NAMES:
        final_drifts = []
        for trial in all_trial_snapshots:
            s = [s for s in trial if s.phase == "evolve" and s.cycle == NUM_CYCLES
                 and s.name == name][0]
            final_drifts.append(s.drift)
        t_stat, p_val = stats.ttest_1samp(final_drifts, 0.0)
        desc = f"One-sample t-test on final drift for {name} (H0: drift=0)"
        result = _ttest_result(t_stat, p_val, desc)
        stat_tests.append(result)
        sig = "*" if result["significant_at_05"] else ""
        print(f"  {name:10s} drift != 0:  t={t_stat:>7.3f}  p={p_val:.6f} {sig}")

    print()

    # 2. Paired t-test on coverage before vs after teaching
    for name in ARCHETYPE_NAMES:
        cov_before = []
        cov_after = []
        for trial in all_trial_snapshots:
            base = [s for s in trial if s.phase == "baseline" and s.name == name][0]
            post_teach = [s for s in trial if s.phase == "teach" and s.cycle == NUM_CYCLES
                          and s.name == name][0]
            cov_before.append(base.coverage)
            cov_after.append(post_teach.coverage)
        t_stat, p_val = stats.ttest_rel(cov_after, cov_before)
        desc = f"Paired t-test coverage before vs after teaching for {name}"
        result = _ttest_result(t_stat, p_val, desc)
        stat_tests.append(result)
        sig = "*" if result["significant_at_05"] else ""
        print(f"  {name:10s} coverage teach effect:  t={t_stat:>7.3f}  p={p_val:.6f} {sig}")

    print()

    # 3. t-test comparing parent vs offspring gene coverage
    breeding_pairs = [
        ("Bold", "Calm", "Sentinel"),
        ("Timid", "Curious", "Scout"),
        ("Bold", "Curious", "Explorer"),
        ("Calm", "Timid", "Guardian"),
    ]
    for pair_idx, (pa_name, pb_name, child_name) in enumerate(breeding_pairs):
        parent_gccs = []
        child_gccs = []
        for trial_breeding in all_trial_breeding:
            br = trial_breeding[pair_idx]
            parent_gccs.append(br["parent_a_gene_cat_coverage"])
            parent_gccs.append(br["parent_b_gene_cat_coverage"])
            child_gccs.append(br["gene_cat_coverage"])
        t_stat, p_val = stats.ttest_ind(child_gccs, parent_gccs)
        desc = (f"Independent t-test offspring ({child_name}) vs parent "
                f"({pa_name},{pb_name}) gene coverage")
        result = _ttest_result(t_stat, p_val, desc)
        stat_tests.append(result)
        sig = "*" if result["significant_at_05"] else ""
        print(f"  {child_name:10s} vs parents gene_cov:  t={t_stat:>7.3f}  p={p_val:.6f} {sig}")

    # ======================================================================
    # Build statistical_summary section
    # ======================================================================
    stat_summary: dict = {
        "n_trials": NUM_TRIALS,
        "seeds": SEEDS,
        "per_archetype": {},
        "breeding": {},
        "tests": stat_tests,
    }

    for name in ARCHETYPE_NAMES:
        arch_stats: dict = {}
        for phase_label, phase, cycle in [
            ("baseline", "baseline", 0),
            ("final_teach", "teach", NUM_CYCLES),
            ("final_evolve", "evolve", NUM_CYCLES),
        ]:
            per_trial = []
            for trial in all_trial_snapshots:
                if phase == "baseline":
                    matches = [s for s in trial if s.phase == phase and s.name == name]
                else:
                    matches = [s for s in trial if s.phase == phase
                               and s.cycle == cycle and s.name == name]
                per_trial.append(matches[0])

            arch_stats[phase_label] = {
                "corpus_size": _mean_std_ci([s.corpus_size for s in per_trial]),
                "drift": _mean_std_ci([s.drift for s in per_trial]),
                "coverage": _mean_std_ci([s.coverage for s in per_trial]),
                "gene_cat_coverage": _mean_std_ci([s.gene_cat_coverage for s in per_trial]),
                "stability": _mean_std_ci([s.stability for s in per_trial]),
            }
        stat_summary["per_archetype"][name] = arch_stats

    for pair_idx, (pa_name, pb_name, child_name) in enumerate(breeding_pairs):
        child_sizes = [tb[pair_idx]["corpus_size"] for tb in all_trial_breeding]
        child_gccs = [tb[pair_idx]["gene_cat_coverage"] for tb in all_trial_breeding]
        child_covs = [tb[pair_idx]["coverage"] for tb in all_trial_breeding]
        child_dist_a = [tb[pair_idx]["dist_to_a"] for tb in all_trial_breeding]
        child_dist_b = [tb[pair_idx]["dist_to_b"] for tb in all_trial_breeding]
        stat_summary["breeding"][child_name] = {
            "parents": [pa_name, pb_name],
            "corpus_size": _mean_std_ci(child_sizes),
            "gene_cat_coverage": _mean_std_ci(child_gccs),
            "coverage": _mean_std_ci(child_covs),
            "dist_to_a": _mean_std_ci(child_dist_a),
            "dist_to_b": _mean_std_ci(child_dist_b),
        }

    # ======================================================================
    # Save JSON
    # ======================================================================
    # Per-trial raw data
    per_trial_data = []
    for trial_idx, seed in enumerate(SEEDS):
        trial_snapshots = all_trial_snapshots[trial_idx]
        trial_breeding = all_trial_breeding[trial_idx]
        per_trial_data.append({
            "trial": trial_idx + 1,
            "seed": seed,
            "snapshots": [
                {
                    "phase": s.phase, "cycle": s.cycle, "name": s.name,
                    "corpus_size": s.corpus_size, "drift": round(s.drift, 4),
                    "coverage": round(s.coverage, 3),
                    "gene_cat_coverage": round(s.gene_cat_coverage, 3),
                    "stability": round(s.stability, 3),
                    "taught": s.taught, "evolved": s.evolved,
                }
                for s in trial_snapshots
            ],
            "breeding": trial_breeding,
        })

    output = {
        "config": {
            "archetypes": ARCHETYPE_NAMES,
            "n_teach_cycles": NUM_CYCLES,
            "teach_per_cycle": TEACH_PER_CYCLE,
            "n_evolve_cycles": NUM_CYCLES,
            "seed": SEED,
            "seeds": SEEDS,
            "n_trials": NUM_TRIALS,
        },
        # Backward-compatible: snapshots and breeding from trial 1 (seed=42)
        "snapshots": [
            {
                "phase": s.phase, "cycle": s.cycle, "name": s.name,
                "corpus_size": s.corpus_size, "drift": round(s.drift, 4),
                "coverage": round(s.coverage, 3),
                "gene_cat_coverage": round(s.gene_cat_coverage, 3),
                "stability": round(s.stability, 3),
                "taught": s.taught, "evolved": s.evolved,
            }
            for s in representative_snapshots
        ],
        "breeding": representative_breeding,
        "per_trial": per_trial_data,
        "statistical_summary": stat_summary,
    }
    json_path = OUT_DIR / "eval8_results.json"

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return super().default(obj)

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {json_path}")

    # ======================================================================
    # Plot (mean +/- std across trials)
    # ======================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Eval 8: Evolution Dynamics (Evolutionary Ecosystem, n={NUM_TRIALS} trials)",
            fontsize=14,
        )

        colors = {"Bold": "#E53935", "Timid": "#7E57C2",
                  "Curious": "#43A047", "Calm": "#1E88E5"}

        # Helper: collect per-trial time series for a metric
        def _collect_series(metric_attr: str, name: str):
            """Return (mean_array, std_array) across trials for the given archetype."""
            # Determine number of snapshots per trial per archetype
            n_steps = len([s for s in all_trial_snapshots[0] if s.name == name])
            values = np.zeros((NUM_TRIALS, n_steps))
            for t_idx, trial in enumerate(all_trial_snapshots):
                arch_snaps = [s for s in trial if s.name == name]
                for s_idx, s in enumerate(arch_snaps):
                    values[t_idx, s_idx] = getattr(s, metric_attr)
            mean = np.mean(values, axis=0)
            std = np.std(values, axis=0, ddof=1) if NUM_TRIALS > 1 else np.zeros_like(mean)
            return mean, std

        # Panel 1: Corpus size over cycles
        ax = axes[0, 0]
        for name in ARCHETYPE_NAMES:
            mean, std = _collect_series("corpus_size", name)
            x_vals = np.arange(len(mean))
            ax.plot(x_vals, mean, "-o", color=colors[name], label=name, markersize=4)
            ax.fill_between(x_vals, mean - std, mean + std,
                            color=colors[name], alpha=0.15)
        ax.set_ylabel("Corpus Size")
        ax.set_xlabel("Development Cycle")
        ax.set_title("Corpus Growth")
        ax.legend(fontsize=8)
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(x=5.5, color="gray", linestyle="--", alpha=0.3)

        # Panel 2: Drift over cycles
        ax = axes[0, 1]
        for name in ARCHETYPE_NAMES:
            mean, std = _collect_series("drift", name)
            x_vals = np.arange(len(mean))
            ax.plot(x_vals, mean, "-o", color=colors[name], label=name, markersize=4)
            ax.fill_between(x_vals, mean - std, mean + std,
                            color=colors[name], alpha=0.15)
        ax.set_ylabel("Cosine Distance from Baseline")
        ax.set_xlabel("Development Cycle")
        ax.set_title("Behavioral Drift")
        ax.legend(fontsize=8)

        # Panel 3: Coverage over cycles
        ax = axes[1, 0]
        for name in ARCHETYPE_NAMES:
            mean, std = _collect_series("coverage", name)
            x_vals = np.arange(len(mean))
            ax.plot(x_vals, mean, "-o", color=colors[name], label=name, markersize=4)
            ax.fill_between(x_vals, mean - std, mean + std,
                            color=colors[name], alpha=0.15)
        ax.set_ylabel("Coverage")
        ax.set_xlabel("Development Cycle")
        ax.set_title("Behavioral Coverage")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

        # Panel 4: Evolved instructions (bar chart, mean +/- std of final state)
        ax = axes[1, 1]
        mean_evolved = []
        std_evolved = []
        mean_taught = []
        std_taught = []
        for name in ARCHETYPE_NAMES:
            ev_vals = []
            ta_vals = []
            for trial in all_trial_snapshots:
                s = [s for s in trial if s.phase == "evolve"
                     and s.cycle == NUM_CYCLES and s.name == name][0]
                ev_vals.append(s.evolved)
                ta_vals.append(s.taught)
            mean_evolved.append(np.mean(ev_vals))
            std_evolved.append(np.std(ev_vals, ddof=1) if NUM_TRIALS > 1 else 0.0)
            mean_taught.append(np.mean(ta_vals))
            std_taught.append(np.std(ta_vals, ddof=1) if NUM_TRIALS > 1 else 0.0)
        x = np.arange(len(ARCHETYPE_NAMES))
        ax.bar(x - 0.15, mean_taught, 0.3, yerr=std_taught, label="Taught",
               color="#FF9800", alpha=0.8, capsize=3)
        ax.bar(x + 0.15, mean_evolved, 0.3, yerr=std_evolved, label="Evolved",
               color="#2196F3", alpha=0.8, capsize=3)
        ax.set_ylabel("Instruction Count")
        ax.set_xticks(x)
        ax.set_xticklabels(ARCHETYPE_NAMES)
        ax.set_title("Taught vs Evolved Instructions")
        ax.legend()

        plt.tight_layout()
        fig_path = OUT_DIR / "eval8_dynamics.png"
        plt.savefig(fig_path, dpi=150)
        print(f"Figure saved to {fig_path}")
        plt.close()

    except ImportError:
        print("matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
