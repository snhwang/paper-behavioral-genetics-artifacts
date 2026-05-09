"""Evolutionary Ecosystem — FastAPI server with dual-path BEAR brain integration.

Merges creature ecosystem's 3D interactive NPC server with spatial evolution's
epoch/weather/energy/food systems. Features:
- Dual-path decision: fast-path (BEAR scores → movement) + slow-path (LLM dialogue)
- 8 gene categories with appearance, skills, stats, behavior profile
- Epoch cycling with food/weather/aggression modulation
- Population statistics tracking
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import random
import sys
from contextlib import asynccontextmanager
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(_HERE.parent.parent.parent / ".env")
except ImportError:
    pass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from bear import Config, EmbeddingBackend, Corpus, Retriever
from bear.llm import LLM
from bear.retriever import Embedder

from .gene_engine import (
    GENE_CATEGORIES,
    AppearanceParams, SkillSet, EntityStats,
    generate_npc_genes, generate_creature_name,
    extract_appearance, extract_skills, extract_stats,
    build_corpus, compute_behavior_profile,
    BreedRequest, BreedResult, breeding_worker,
)
from .sim import (
    Creature, World, WorldItem, FoodItem, Predator,
    WORLD_W, WORLD_H, BREED_COOLDOWN, ENERGY_MAX,
    MAX_AGE_MIN, MAX_AGE_MAX, MAX_POPULATION,
    PREDATOR_SPAWN_INTERVAL,
    tick, distance, check_fights,
    VALID_ANIMATIONS, VALID_MOODS, VALID_EFFECTS,
)
from .epochs import EPOCHS
from .brain import CreatureAgent, BrainEngine
from .stats import PopulationTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

world:          World | None       = None
llm:            LLM | None         = None
embedder:       Embedder | None    = None
bear_config:    Config | None      = None
brain_engine:   BrainEngine | None = None
pop_tracker:    PopulationTracker  = PopulationTracker()
breed_q:        asyncio.Queue      = asyncio.Queue()
birth_q:        asyncio.Queue      = asyncio.Queue()
connected:      list[WebSocket]    = []
args_ns:        argparse.Namespace | None = None
rng:            random.Random      = random.Random(42)

CLIENT_DIR = _HERE.parent / "client"

# ---------------------------------------------------------------------------
# Creature factory
# ---------------------------------------------------------------------------

BATCH_SIZE = 4


async def _make_creature(
    cid: str,
    genes: dict[str, str],
    name: str,
    embedder: Embedder,
    bear_config: Config,
    rng: random.Random,
    generation: int = 0,
    parents: tuple[str, str] | None = None,
    spawn_x: float | None = None,
    spawn_y: float | None = None,
) -> Creature:
    """Build a Creature from genes."""
    appearance = extract_appearance(genes, embedder)
    skills     = extract_skills(genes, embedder)
    stats      = extract_stats(genes, embedder)
    corpus     = build_corpus(name, genes)
    behavior   = compute_behavior_profile(corpus, bear_config, shared_embedder=embedder)

    x = spawn_x if spawn_x is not None else rng.uniform(1.0, WORLD_W - 1.0)
    y = spawn_y if spawn_y is not None else rng.uniform(1.0, WORLD_H - 1.0)

    if generation == 0:
        appearance.primary_hue = 45.0

    c = Creature(
        id               = cid,
        name             = name,
        x                = x,
        y                = y,
        genes            = genes,
        appearance       = appearance,
        skills           = skills,
        stats            = stats,
        behavior_profile = behavior,
        happiness        = rng.uniform(68, 88),
        heading          = rng.uniform(0, 2 * math.pi),
        generation       = generation,
        parents          = parents,
        corpus           = corpus,
        max_age          = rng.uniform(MAX_AGE_MIN, MAX_AGE_MAX),
        hp               = 100.0,
        energy           = rng.uniform(70, 100),
    )
    c.retriever = _make_retriever(corpus)
    return c


def _make_retriever(corpus: Corpus) -> Retriever:
    """Build retriever, applying diploid expression if needed."""
    from bear.evolution import express as bear_express
    from bear.models import LocusRegistry, GeneLocus, Dominance
    from .gene_engine import GENE_CATEGORIES

    use_corpus = corpus
    if world and world.ploidy.startswith("diploid"):
        dom_map = {
            "diploid_dominant":   Dominance.DOMINANT,
            "diploid_codominant": Dominance.CODOMINANT,
        }
        dom = dom_map.get(world.ploidy, Dominance.DOMINANT)
        registry = LocusRegistry(loci=[
            GeneLocus(name=cat, position=i, dominance=dom)
            for i, cat in enumerate(GENE_CATEGORIES)
        ])
        expressed = bear_express(corpus, registry, locus_key="gene_category")
        use_corpus = Corpus()
        for inst in expressed:
            use_corpus.add(inst)

    retriever = Retriever(use_corpus, config=bear_config)
    retriever._embedder = embedder
    retriever.build_index()
    return retriever


def _unique_name(name: str, existing: set[str], rng: random.Random) -> str:
    if name not in existing:
        return name
    suffixes = list("ABCDEFGHJKLMNPQRSTVWXYZ")
    rng.shuffle(suffixes)
    for s in suffixes:
        candidate = f"{name} {s}"
        if candidate not in existing:
            return candidate
    i = 2
    while f"{name}{i}" in existing:
        i += 1
    return f"{name}{i}"


async def _add_one_creature() -> bool:
    try:
        genes = await generate_npc_genes(llm, rng)
        name  = await generate_creature_name(llm, genes, rng)
        existing_names = {c.name for c in world.creatures.values()}
        name  = _unique_name(name, existing_names, rng)
        cid   = world.next_id()
        creature = await _make_creature(cid, genes, name, embedder, bear_config, rng)
        world.creatures[cid] = creature

        retriever = _make_retriever(creature.corpus)
        agent     = CreatureAgent(cid, retriever, llm, rng)
        brain_engine.agents[cid] = agent

        logger.info("  %s (%s): body=%.2f hue=%.0f speed=%.2f",
                    name, cid,
                    creature.appearance.body_radius,
                    creature.appearance.primary_hue,
                    creature.skills.speed_base)
        return True
    except Exception as e:
        logger.error("Failed to create creature: %s", e, exc_info=True)
        return False


def _spawn_items(world: World, rng: random.Random) -> None:
    configs = [
        ("flower", 6),
        ("tree",   4),
        ("rock",   5),
        ("food",   5),
        ("ball",   1),
    ]
    for item_type, count in configs:
        for i in range(count):
            iid = f"{item_type}_{i + 1}"
            x   = rng.uniform(1.5, WORLD_W - 1.5)
            y   = rng.uniform(1.5, WORLD_H - 1.5)
            world.items[iid] = WorldItem(id=iid, type=item_type, x=x, y=y)
    logger.info("Spawned %d world items", sum(c for _, c in configs))
    # Pre-spawn food so creatures have something to eat immediately
    for _ in range(15):
        world.food.append(FoodItem(
            x=rng.uniform(1.0, WORLD_W - 1.0),
            y=rng.uniform(1.0, WORLD_H - 1.0),
        ))


async def _populate(target: int) -> None:
    world.paused = True
    logger.info("Generating %d creatures...", target)

    added = 0
    first = True
    for i in range(target):
        ok = await _add_one_creature()
        if ok:
            added += 1
            if first:
                world.paused = False
                first = False
            logger.info("  %d/%d creatures ready", added, target)

    if first:
        world.paused = False
    logger.info("Population complete (%d/%d creatures)", added, target)


# ---------------------------------------------------------------------------
# Background loops
# ---------------------------------------------------------------------------


# ── Sim data logging ────────────────────────────────────────────────────────
_epoch_snapshots: list[dict] = []
_birth_log:       list[dict] = []
_snapshot_log:    list[dict] = []
_action_log:      list[dict] = []   # slow-path decision events
_death_log:       list[dict] = []   # per-creature death records
_pending_parent_genes: dict = {}  # child_id -> (pa_genes, pb_genes)


def _avg_behavior(creatures) -> dict:
    """Mean bear_strength profile across creatures."""
    QUERIES = [
        ("food_seeking",  "hungry foraging find food eat",           ["food"]),
        ("combat",        "fight aggression territorial combat",     ["combat"]),
        ("survival",      "survive starvation endurance resilience", ["survival"]),
        ("stealth",       "hide conceal stealth avoid detection",    ["stealth"]),
        ("breeding",      "mate reproduce offspring eager breed",    []),
    ]
    result = {}
    for name, query, tags in QUERIES:
        vals = [c.bear_strength(query, tags) for c in creatures
                if c.retriever is not None]
        result[name] = round(float(sum(vals)/len(vals)), 4) if vals else 0.3
    return result


def _capture_epoch_snapshot(w) -> None:
    creatures = list(w.creatures.values())
    _epoch_snapshots.append({
        "tick":       w.tick_count,
        "epoch":      w.epoch.name,
        "population": len(creatures),
        "births":     w.total_births,
        "avg_behavior": _avg_behavior(creatures),
    })


def _capture_snapshot(w) -> None:
    """Periodic population snapshot — every N ticks."""
    creatures = list(w.creatures.values())
    _snapshot_log.append({
        "tick":       w.tick_count,
        "epoch":      w.epoch.name,
        "population": len(creatures),
        "births":     w.total_births,
        "deaths":     w.total_deaths,
        "avg_behavior": _avg_behavior(creatures),
        "per_creature": [
            {"id": c.id, "name": c.name, "generation": c.generation,
             "genes": dict(c.genes), "behavior": _avg_behavior([c])}
            for c in creatures if c.retriever is not None
        ],
    })


def _log_birth(result) -> None:
    """Record every birth with parent and child gene data."""
    _birth_log.append({
        "tick":           getattr(world, "tick_count", 0),
        "epoch":          getattr(world, "epoch", None) and world.epoch.name,
        "child_name":     result.child_name,
        "generation":     result.generation,
        "parent_a":       result.parent_a_name,
        "parent_b":       result.parent_b_name,
        "child_genes":    dict(result.genes),
        "parent_a_genes": dict(getattr(world.creatures.get(result.parent_a_name
                              if hasattr(result, "parent_a_name") else ""), "genes", {})),
        "parent_b_genes": dict(getattr(world.creatures.get(result.parent_b_name
                              if hasattr(result, "parent_b_name") else ""), "genes", {})),
    })


# Chunked output: when --chunk-size N is set, the simulation rotates the
# output file every N ticks. Each chunk gets a numbered suffix (e.g.
# foo_01.json, foo_02.json) and accumulator lists are cleared at chunk
# boundaries so each file stays self-contained and bounded in size.
_chunk_index: int = 1


def _chunked_output_path(base_output: str) -> str:
    """Return the current chunk's output path, or *base_output* if no chunking."""
    import os as _os
    if getattr(args_ns, "chunk_size", None):
        base, ext = _os.path.splitext(base_output)
        return f"{base}_{_chunk_index:02d}{ext}"
    return base_output


async def _maybe_finalize_chunk(w) -> None:
    """At a chunk boundary, flush current data to disk then clear accumulators.

    Must be awaited: previously this used asyncio.create_task for autosave and
    cleared accumulators synchronously, which raced — the scheduled autosave
    saw cleared lists by the time it ran, dropping all events accumulated
    since the previous autosave (~2000 ticks of births/deaths/actions per
    chunk boundary).
    """
    global _chunk_index
    cs = getattr(args_ns, "chunk_size", None)
    if not cs or w.tick_count == 0:
        return
    if w.tick_count % cs != 0:
        return
    # Flush THIS chunk's accumulators to its file before clearing.
    if getattr(args_ns, "output", None):
        await _auto_save(w)
    _birth_log.clear()
    _death_log.clear()
    _action_log.clear()
    _snapshot_log.clear()
    _epoch_snapshots.clear()
    _chunk_index += 1
    logger.info("Chunk boundary at tick %d — finalised chunk %02d, advancing to %02d",
                w.tick_count, _chunk_index - 1, _chunk_index)


async def _save_sim_log() -> None:
    """Save all collected sim data to log file."""
    if not world:
        return
    from examples.evolutionary_ecosystem.eval.harness import gene_diversity_mean
    creatures = list(world.creatures.values())
    log = {
        "metadata": {
            "seed":       getattr(args_ns, "seed", 42),
            "epoch":      getattr(args_ns, "lock_epoch", None),
            "n_ticks":    getattr(args_ns, "ticks", world.tick_count),
            "n_creatures": getattr(args_ns, "creatures", 4),
            "recombination": getattr(args_ns, "recombination", "locus"),
        },
        "final": {
            "population":   len(creatures),
            "total_births": world.total_births,
            "total_deaths": world.total_deaths,
            "max_generation": max((c.generation for c in creatures), default=0),
            "gene_diversity": gene_diversity_mean(creatures),
            "avg_behavior": _avg_behavior(creatures),
        },
        "birth_log":      _birth_log,
        "death_log":      _death_log,
        "action_log":     _action_log,
        "snapshots":      _snapshot_log,
        "epoch_snapshots": _epoch_snapshots,
    }
    if getattr(args_ns, "chunk_size", None):
        log["metadata"]["chunk_index"] = _chunk_index
        log["metadata"]["chunk_size"] = args_ns.chunk_size
    out = getattr(args_ns, "output", None)
    if not out:
        seed = log["metadata"]["seed"]
        epoch = log["metadata"]["epoch"] or "free"
        out = f"sim_log_seed{seed}_{epoch}.json"
    out_path = _chunked_output_path(out)
    Path(out_path).write_text(json.dumps(log, indent=2))
    logger.info("Sim log saved to %s", out_path)
    print(f"Saved to {out_path} — {len(_birth_log)} births, {len(_snapshot_log)} snapshots")


async def _auto_save(w) -> None:
    """Periodic auto-save during live sim — survives crashes."""
    from pathlib import Path
    try:
        from examples.evolutionary_ecosystem.eval.harness import gene_diversity_mean
        creatures = list(w.creatures.values())
        log = {
            "tick":         w.tick_count,
            "epoch":        w.epoch.name,
            "population":   len(creatures),
            "total_births": w.total_births,
            "total_deaths": w.total_deaths,
            "gene_diversity": gene_diversity_mean(creatures),
            "avg_behavior": _avg_behavior(creatures),
            "birth_log":    _birth_log,
            "death_log":    _death_log,
            "action_log":   _action_log,
            "snapshots":    _snapshot_log,
            "epoch_snapshots": _epoch_snapshots,
        }
        if getattr(args_ns, "chunk_size", None):
            log["chunk_index"] = _chunk_index
            log["chunk_size"] = args_ns.chunk_size
        out_base = getattr(args_ns, "output", None) or "sim_autosave.json"
        out_path = _chunked_output_path(out_base)
        Path(out_path).write_text(json.dumps(log, indent=2))
        logger.info("Auto-saved sim log to %s (%d births in this chunk)", out_path, len(_birth_log))
    except Exception as e:
        logger.warning("Auto-save failed: %s", e)


# Keep _save_eval_results as alias for backwards compatibility
_save_eval_results = _save_sim_log


async def _simulation_loop(tick_rate: int) -> None:
    dt = 1.0 / tick_rate
    while True:
        try:
            if world and not world.paused:
                effective_dt = dt * world.speed_mult
                tick(world, rng, effective_dt)

                # Clean up retired creatures' brain agents + flush death log
                if world.retired_ids:
                    for rid in world.retired_ids:
                        if brain_engine and rid in brain_engine.agents:
                            del brain_engine.agents[rid]
                    world.retired_ids.clear()
                if world.death_log_pending:
                    _death_log.extend(world.death_log_pending)
                    world.death_log_pending.clear()

                # Check birth queue
                await _process_births()

                # Epoch boundary snapshot for eval mode
                if world.epoch_timer == 0 and world.tick_count > 0:
                    _capture_epoch_snapshot(world)
                if world.tick_count % 500 == 0:
                    _capture_snapshot(world)
                # Slow tick loop for blend to prevent extinction
                if getattr(args_ns, "recombination", "locus") == "blend":
                    await asyncio.sleep(0.5)  # 500ms = ~2 ticks/sec, matches LLM breed speed
                # Auto-save every 2000 ticks so data survives crashes (opt-in via --output)
                if (world.tick_count % 2000 == 0 and world.tick_count > 0
                        and getattr(args_ns, "output", None)):
                    asyncio.create_task(_auto_save(world))

                # Chunk rotation: at every chunk_size boundary, flush the
                # current chunk to disk then clear accumulators so the next
                # chunk file starts fresh and stays bounded in size.
                await _maybe_finalize_chunk(world)

                # Tick limit for eval mode
                max_ticks = getattr(args_ns, 'ticks', None)
                if max_ticks and world.tick_count >= max_ticks:
                    logger.info("Tick limit %d reached -- saving results and exiting", max_ticks)
                    await _save_eval_results()
                    import os
                    os._exit(0)
        except SystemExit:
            raise
        except Exception as e:
            logger.error("Simulation loop error: %s", e, exc_info=True)
        await asyncio.sleep(dt)


async def _process_births() -> None:
    while not birth_q.empty():
        if len(world.creatures) >= MAX_POPULATION:
            logger.warning("Population cap (%d) — discarding pending birth", MAX_POPULATION)
            await _broadcast(json.dumps({
                "type": "warning",
                "text": f"Population cap reached ({MAX_POPULATION}) — breeding suspended",
            }))
            # Drain remaining births
            while not birth_q.empty():
                try: birth_q.get_nowait()
                except asyncio.QueueEmpty: break
            return
        try:
            result: BreedResult = birth_q.get_nowait()
        except asyncio.QueueEmpty:
            break

        cid = result.child_id
        creature = Creature(
            id               = cid,
            name             = result.child_name,
            x                = result.spawn_x,
            y                = result.spawn_y,
            genes            = result.genes,
            appearance       = result.appearance,
            skills           = result.skills,
            stats            = result.stats,
            behavior_profile = result.behavior,
            happiness        = 60.0,
            generation       = result.generation,
            parents          = (result.parent_a_name, result.parent_b_name),
            corpus           = result.corpus,
            max_age          = rng.uniform(MAX_AGE_MIN, MAX_AGE_MAX),
            hp               = 100.0,
            energy           = 80.0,
        )
        world.creatures[cid] = creature
        world.total_births += 1
        # Use cached parent genes (captured at breed request time)
        _cached = _pending_parent_genes.pop(cid, None)
        pa_genes = _cached[0] if _cached else {}
        pb_genes = _cached[1] if _cached else {}
        _birth_log.append({
            "tick":       world.tick_count,
            "epoch":      world.epoch.name,
            "child_name": result.child_name,
            "generation": result.generation,
            "parent_a":   result.parent_a_name,
            "parent_b":   result.parent_b_name,
            "child_genes": dict(result.genes),
            "pa_genes":   pa_genes,
            "pb_genes":   pb_genes,
        })

        retriever = _make_retriever(result.corpus)
        creature.retriever = retriever
        agent = CreatureAgent(cid, retriever, llm, rng)
        brain_engine.agents[cid] = agent

        logger.info("Born: %s (gen %d, parents: %s x %s)",
                    result.child_name, result.generation,
                    result.parent_a_name, result.parent_b_name)

        msg = json.dumps({
            "type":     "birth",
            "creature": creature.to_dict(),
        })
        await _broadcast(msg)


async def _breed_dispatch_loop() -> None:
    while True:
        if world:
            while not world.breed_queue.empty():
                try:
                    item = world.breed_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if isinstance(item, tuple) and item[0] == "breed":
                    _, id_a, id_b = item
                    a = world.creatures.get(id_a)
                    b = world.creatures.get(id_b)
                    if not a or not b:
                        dead = [x for x, c in [(id_a, a), (id_b, b)] if not c]
                        logger.warning("Breed dropped — parent(s) dead: %s", dead)
                    if a and b:
                        child_id = world.next_id()
                        existing_names = {c.name for c in world.creatures.values()}
                        child_name = _unique_name(
                            f"{a.name[:4]}{b.name[-3:]}", existing_names, rng
                        )
                        req = BreedRequest(
                            parent_a_genes   = a.genes,
                            parent_b_genes   = b.genes,
                            parent_a_name    = a.name,
                            parent_b_name    = b.name,
                            parent_a_corpus  = a.corpus,
                            parent_b_corpus  = b.corpus,
                            parent_a_appear  = a.appearance,
                            parent_b_appear  = b.appearance,
                            parent_a_fitness = a.happiness,
                            parent_b_fitness = b.happiness,
                            child_name       = child_name,
                            child_id         = child_id,
                            spawn_x          = (a.x + b.x) / 2,
                            spawn_y          = (a.y + b.y) / 2,
                            generation       = max(a.generation, b.generation) + 1,
                            recombination    = world.recombination,
                            ploidy           = world.ploidy,
                        )
                        # Cache parent genes for birth logging
                        _pending_parent_genes[child_id] = (
                            dict(a.genes), dict(b.genes))
                        await breed_q.put(req)
                        if breed_q.qsize() > 1:
                            logger.info("Breed queue depth: %d", breed_q.qsize())
        await asyncio.sleep(0.1)


async def _broadcast(data: str) -> None:
    stale: list[WebSocket] = []
    for ws in connected:
        try:
            await ws.send_text(data)
        except Exception:
            stale.append(ws)
    for ws in stale:
        if ws in connected:
            connected.remove(ws)


async def _ws_broadcast_loop(ws_fps: int) -> None:
    interval = 1.0 / ws_fps
    stats_interval = 20  # update pop stats every 20 broadcasts
    counter = 0
    while True:
        if world and connected:
            state = world.get_state()

            # Add population stats periodically
            counter += 1
            if counter % stats_interval == 0:
                pop_stats = pop_tracker.update(world)
                state["pop_stats"] = pop_stats.to_dict()

            await _broadcast(json.dumps(state))
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# LLM auto-detection
# ---------------------------------------------------------------------------

def _probe_openai_compat(base_url: str) -> str | None:
    import json, urllib.request
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=2) as r:
            data = json.loads(r.read())
            models = [m["id"] for m in data.get("data", []) if m.get("id")]
            return models[0] if models else "local"
    except Exception:
        return None


def _auto_llm(base_url: str | None = None, model: str | None = None) -> LLM:
    from bear.config import LLMBackend

    if base_url:
        detected = model or _probe_openai_compat(base_url) or "local"
        logger.info("Using explicit base URL %s, model=%s", base_url, detected)
        return LLM(backend=LLMBackend.OPENAI, model=detected, base_url=base_url)

    # Route Anthropic model names to the Anthropic backend directly
    if model and ("claude" in model.lower() or "anthropic" in model.lower()):
        logger.info("Using Anthropic backend for model=%s", model)
        return LLM(backend=LLMBackend.ANTHROPIC, model=model)

    lms_url = "http://localhost:1234/v1"
    lms_model = _probe_openai_compat(lms_url)
    if lms_model:
        detected = model or lms_model
        logger.info("Auto-detected LM Studio at %s, model=%s", lms_url, detected)
        return LLM(backend=LLMBackend.OPENAI, model=detected, base_url=lms_url)

    if model:
        # Pass explicit model to auto() by trying OpenAI first
        logger.info("Using OpenAI backend for model=%s", model)
        return LLM(backend=LLMBackend.OPENAI, model=model)

    return LLM.auto()


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(application: FastAPI):
    global world, llm, embedder, bear_config, brain_engine, rng

    args = args_ns or argparse.Namespace(
        creatures=4, seed=42, embedding_model="BAAI/bge-base-en-v1.5",
        tick_rate=10, ws_fps=10, model=None, base_url=None,
    )
    rng = random.Random(args.seed)

    bear_config = Config(
        embedding_backend=EmbeddingBackend.NUMPY,
        embedding_model=args.embedding_model,
        mandatory_tags=[],
    )

    base_url = getattr(args, "base_url", None) or None
    model    = getattr(args, "model", None) or None
    llm      = _auto_llm(base_url=base_url, model=model)
    embedder = Embedder(bear_config.embedding_model)
    logger.info("LLM: %s/%s  Embedder: %s",
                llm.backend_type.value, llm.model or "auto", bear_config.embedding_model)
    embedder.embed_single("warmup")
    logger.info("Embedding model loaded")

    recomb = getattr(args, "recombination", "locus")
    ploidy = getattr(args, "ploidy", "haploid")
    world = World(
        breed_queue    = asyncio.Queue(),
        birth_queue    = birth_q,
        predator       = Predator(
            x        = WORLD_W / 2,
            y        = WORLD_H / 2,
            active   = False,
        ),
        epoch          = EPOCHS[0],
        recombination  = recomb,
        ploidy         = ploidy,
    )
    logger.info("Breeding: recombination=%s, ploidy=%s", recomb, ploidy)
    world.predator.spawn_at = world.t + PREDATOR_SPAWN_INTERVAL
    _spawn_items(world, rng)

    brain_engine = BrainEngine(
        world       = world,
        agents      = {},
        llm         = llm,
        rng         = rng,
        bear_config = bear_config,
        tick_interval = 0.5,
    )
    brain_engine._action_log = _action_log  # enable action tag logging

    # Lock epoch if requested
    if getattr(args, 'lock_epoch', None):
        locked = next((e for e in EPOCHS if e.name == args.lock_epoch), None)
        if locked:
            world.epoch = locked
            world.epoch_locked = True
            logger.info("Epoch locked to: %s", locked.name)
        else:
            logger.warning("Unknown epoch: %s", args.lock_epoch)

    # Apply BEAR Off condition if requested via --bear-disabled
    if getattr(args, 'bear_disabled', False):
        world.bear_disabled = True
        brain_engine.bear_disabled = True
        world.autonomous_breeding = True
        logger.info("BEAR disabled (eval BEAR Off condition)")

    tasks = [
        asyncio.create_task(_populate(args.creatures)),
        asyncio.create_task(_simulation_loop(args.tick_rate)),
        asyncio.create_task(brain_engine.run()),
        asyncio.create_task(_breed_dispatch_loop()),
        *[asyncio.create_task(
            breeding_worker(llm, embedder, breed_q, birth_q,
                            random.Random(rng.randint(0, 2**31)), bear_config)
        ) for _ in range(3)],
        asyncio.create_task(_ws_broadcast_loop(args.ws_fps)),
    ]

    yield

    for t in tasks:
        t.cancel()


app = FastAPI(title="Evolutionary Ecosystem", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def index():
    return FileResponse(str(CLIENT_DIR / "index.html"))


@app.get("/main.js")
async def main_js():
    return FileResponse(str(CLIENT_DIR / "main.js"), media_type="application/javascript")


@app.get("/style.css")
async def style_css():
    return FileResponse(str(CLIENT_DIR / "style.css"), media_type="text/css")


@app.get("/health")
async def health():
    n = len(world.creatures) if world else 0
    return {
        "status":    "ok",
        "creatures": n,
        "brain":     brain_engine.stats if brain_engine else {},
    }


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    connected.append(ws)
    logger.info("Client connected (%d total)", len(connected))

    if world:
        await ws.send_text(json.dumps(world.get_state()))

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")
            if msg_type not in ("select", "deselect", ""):
                logger.info("WS msg: %s", msg_type)

            if msg_type == "pause":
                if world:
                    world.paused = True
            elif msg_type == "resume":
                if world:
                    world.paused = False
            elif msg_type == "toggle_pause":
                if world:
                    world.paused = not world.paused
            elif msg_type == "speed":
                if world:
                    world.speed_mult = max(0.25, min(5.0, float(msg.get("value", 1.0))))
            elif msg_type == "poke":
                cid = msg.get("id")
                if world and cid and cid in world.creatures:
                    world.creatures[cid].happiness = min(100.0,
                        world.creatures[cid].happiness + 15)
                    world.creatures[cid].energy = min(ENERGY_MAX,
                        world.creatures[cid].energy + 20)
                    world.creatures[cid].effect = "sparkles"
            elif msg_type == "bear_set":
                if world and brain_engine:
                    enabled = bool(msg.get("enabled", True))
                    world.bear_disabled = not enabled
                    brain_engine.bear_disabled = world.bear_disabled
                    logger.info("BEAR %s", "disabled" if world.bear_disabled else "enabled")
                    await _broadcast(json.dumps(world.get_state()))
            elif msg_type == "amplifier":
                if world:
                    param = msg.get("param", "")
                    value = max(0.0, min(3.0, float(msg.get("value", 1.0))))
                    if param in world.amplifiers:
                        world.amplifiers[param] = value
                        logger.info("AMP %s = %.1f×", param, value)
            elif msg_type == "set_breeding":
                if world:
                    recomb = msg.get("recombination")
                    ploidy = msg.get("ploidy")
                    if recomb in ("locus", "blend", "splice"):
                        world.recombination = recomb
                    if ploidy in ("haploid", "diploid_dominant", "diploid_codominant"):
                        world.ploidy = ploidy
                    logger.info("Breeding mode: %s / %s", world.recombination, world.ploidy)
            elif msg_type == "sim_param":
                if world:
                    param = msg.get("param", "")
                    value = float(msg.get("value", 0))
                    if param in world.sim_params:
                        world.sim_params[param] = value
                        logger.info("SIM %s = %.2f", param, value)
            elif msg_type == "autonomous_breeding":
                if world:
                    world.autonomous_breeding = bool(msg.get("enabled", False))
                    logger.info("Autonomous breeding %s", "ON" if world.autonomous_breeding else "OFF")
            elif msg_type == "auto_regulate":
                if world:
                    world.auto_regulate = bool(msg.get("enabled", False))
                    target = msg.get("target")
                    if target is not None:
                        world.auto_regulate_target = int(target)
                    logger.info("Auto-regulate %s (target=%d)",
                                "ON" if world.auto_regulate else "OFF",
                                world.auto_regulate_target)
            elif msg_type == "restart":
                if world:
                    try:
                        logger.info("RESTART requested")
                        world.paused = True
                        # Preserve settings
                        saved = (world.recombination, world.ploidy, world.amplifiers.copy())
                        # Clear everything
                        world.creatures.clear()
                        world.food.clear()
                        world.items.clear()
                        world.retired_ids.clear()
                        world.challenge_queue.clear()
                        world.total_births = 0
                        world.total_deaths = 0
                        world.tick_count = 0
                        world.epoch_index = 0
                        world.epoch = EPOCHS[0]
                        world.epoch_timer = 0
                        world.weather = "mild"
                        world.weather_timer = 0
                        world.t = __import__("time").time()
                        world._id_counter = 0
                        world.recombination, world.ploidy, world.amplifiers = saved
                        # Drain queues safely
                        for q in (world.breed_queue, breed_q, birth_q):
                            if q:
                                while not q.empty():
                                    try: q.get_nowait()
                                    except Exception: break
                        if world.predator:
                            world.predator.active = False
                            world.predator.spawn_at = world.t + PREDATOR_SPAWN_INTERVAL
                        if brain_engine:
                            brain_engine.agents.clear()
                        _spawn_items(world, rng)
                        # Broadcast empty state so client clears meshes
                        await _broadcast(json.dumps(world.get_state()))
                        # Spawn new creatures
                        n = getattr(args_ns, "creatures", 6) or 6
                        asyncio.create_task(_populate(n))
                        logger.info("RESTART complete — spawning %d creatures", n)
                    except Exception as e:
                        logger.error("RESTART failed: %s", e, exc_info=True)
                        world.paused = False
            elif msg_type == "select":
                pass  # Client-side selection
            elif msg_type == "deselect":
                pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket error: %s", e)
    finally:
        if ws in connected:
            connected.remove(ws)
        logger.info("Client disconnected (%d remaining)", len(connected))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _run_headless(args: argparse.Namespace) -> None:
    """Run the full sim headlessly for N ticks and write a results JSON."""
    import json, time as _time
    from pathlib import Path
    from examples.evolutionary_ecosystem.server.gene_engine import NullBehaviorProfile
    from examples.evolutionary_ecosystem.eval.harness import gene_diversity_mean

    rng_h = random.Random(args.seed)
    bear_cfg = Config(
        embedding_backend=EmbeddingBackend.NUMPY,
        embedding_model=getattr(args, "embedding_model", "BAAI/bge-base-en-v1.5"),
        mandatory_tags=[],
    )
    embedder_h = Embedder(bear_cfg.embedding_model)
    embedder_h.embed_single("warmup")

    llm_h = _auto_llm(base_url=getattr(args, "base_url", None),
                      model=getattr(args, "model", None))

    # _make_retriever reads module-level `embedder`, `bear_config`, and `world` globals
    global embedder, bear_config, world, llm, brain_engine
    embedder = embedder_h
    bear_config = bear_cfg
    llm = llm_h

    bq = asyncio.Queue()
    biq = asyncio.Queue()
    # Apply lock_epoch if specified
    _locked_epoch = EPOCHS[0]
    if getattr(args, "lock_epoch", None):
        _match = next((e for e in EPOCHS if e.name == args.lock_epoch), None)
        if _match:
            _locked_epoch = _match
            logger.info("Locking epoch to: %s", _match.name)
        else:
            logger.warning("Unknown epoch %s, using default", args.lock_epoch)
    w = World(
        breed_queue   = bq,
        birth_queue   = biq,
        predator      = Predator(x=WORLD_W/2, y=WORLD_H/2, active=False),
        epoch         = _locked_epoch,
        recombination = getattr(args, "recombination", "locus"),
        ploidy        = getattr(args, "ploidy", "haploid"),
    )
    world = w
    # Prevent epoch transitions if locked
    if getattr(args, "lock_epoch", None):
        w.epoch_timer = 10**9  # effectively never transitions
    w.predator.spawn_at = w.t + PREDATOR_SPAWN_INTERVAL
    # Enable autonomous breeding when brain is off (locus recombination)
    # or when BEAR is disabled — proximity-based trigger handles breeding
    _no_brain = getattr(args, "recombination", "locus") != "blend"
    w.autonomous_breeding = args.bear_disabled or _no_brain

    _spawn_items(w, rng_h)

    brain_h = BrainEngine(
        world         = w,
        agents        = {},
        llm           = None if args.bear_disabled else llm_h,
        rng           = rng_h,
        bear_config   = bear_cfg,
        tick_interval = 0.5,
    )
    brain_engine = brain_h

    breed_tasks = [
        asyncio.create_task(
            breeding_worker(
                None if args.bear_disabled else llm_h,
                embedder_h, bq, biq,
                random.Random(rng_h.randint(0, 2**31)), bear_cfg,
            )
        ) for _ in range(3)
    ]
    # Skip brain LLM decisions for locus recombination — fast path handles everything
    _use_brain = getattr(args, "recombination", "locus") == "blend"
    brain_task = asyncio.create_task(brain_h.run()) if _use_brain else asyncio.create_task(asyncio.sleep(0))

    # Populate
    await _headless_populate(w, brain_h, args.creatures, rng_h, bear_cfg,
                             embedder_h, args.bear_disabled, llm=llm_h)

    if args.bear_disabled:
        for c in w.creatures.values():
            c.behavior_profile = NullBehaviorProfile()

    n_ticks = getattr(args, "ticks", 30000)
    dt = 0.1
    t_start = _time.perf_counter()
    snapshots = []

    for t in range(n_ticks):
        tick(w, rng_h, dt)
        # drain births
        while not biq.empty():
            try:
                child = biq.get_nowait()
                if args.bear_disabled:
                    child.behavior_profile = NullBehaviorProfile()
                else:
                    from bear.retriever import Retriever
                    r = Retriever(child.corpus, config=bear_cfg)
                    r.build_index()
                    brain_h.agents[child.id] = CreatureAgent(
                        creature_id=child.id, retriever=r,
                        llm=llm_h, rng=random.Random(rng_h.randint(0, 2**31)),
                    )
                if len(w.creatures) < MAX_POPULATION:
                    w.creatures[child.id] = child
                    w.total_births += 1
                    _birth_log.append({
                        "tick": t,
                        "epoch": w.epoch.name,
                        "child_name": child.name,
                        "generation": child.generation,
                        "parent_a": child.parents[0] if child.parents else None,
                        "parent_b": child.parents[1] if child.parents else None,
                        "child_genes": dict(child.genes),
                        "parent_a_genes": dict(w.creatures.get(
                            child.parents[0], child).genes)
                            if child.parents and child.parents[0] in w.creatures else {},
                        "parent_b_genes": dict(w.creatures.get(
                            child.parents[1], child).genes)
                            if child.parents and child.parents[1] in w.creatures else {},
                    })
            except asyncio.QueueEmpty:
                break
        # clean up dead agents
        for rid in list(w.retired_ids):
            brain_h.agents.pop(rid, None)
        w.retired_ids.clear()
        await asyncio.sleep(0)  # yield to event loop so breed workers can process

        if t % 1000 == 0 and t > 0:
            print(f"  tick {t:6d}: pop={len(w.creatures)} births={w.total_births} deaths={w.total_deaths}", flush=True)
        if t % 500 == 0:
            _clist = list(w.creatures.values())
            snapshots.append({
                "tick":           t,
                "epoch":          w.epoch.name,
                "population":     len(_clist),
                "births":         w.total_births,
                "deaths":         w.total_deaths,
                "gene_diversity": gene_diversity_mean(_clist),
                "avg_behavior":   _avg_behavior(_clist),
            })

    elapsed = _time.perf_counter() - t_start

    brain_task.cancel()
    for bt in breed_tasks:
        bt.cancel()
    await asyncio.gather(brain_task, *breed_tasks, return_exceptions=True)

    creatures = list(w.creatures.values())
    # Per-gene diversity for detailed analysis
    from examples.evolutionary_ecosystem.eval.harness import compute_per_gene_diversity, compute_hausdorff_diversity
    per_gene = compute_per_gene_diversity(creatures) if creatures else {}
    hausdorff = compute_hausdorff_diversity(creatures) if len(creatures) >= 2 else {}
    hausdorff_mean = float(sum(hausdorff.values()) / len(hausdorff)) if hausdorff else 0.0

    results = {
        "metadata": {
            "seed":          args.seed,
            "epoch":         getattr(args, "lock_epoch", None),
            "recombination": getattr(args, "recombination", "locus"),
            "n_ticks":       n_ticks,
            "n_creatures":   args.creatures,
            "elapsed_seconds": round(elapsed, 1),
        },
        "final": {
            "population":    len(creatures),
            "total_births":  w.total_births,
            "total_deaths":  w.total_deaths,
            "max_generation": max((c.generation for c in creatures), default=0),
            "gene_diversity": gene_diversity_mean(creatures),
            "per_gene_diversity": per_gene,
            "hausdorff":     hausdorff_mean,
            "avg_behavior":  _avg_behavior(creatures),
        },
        "birth_log":     _birth_log,
        "snapshots":     snapshots,
        "epoch_snapshots": _epoch_snapshots,
    }

    out = args.output or f"headless_seed{args.seed}_{'on' if not args.bear_disabled else 'off'}.json"
    Path(out).write_text(json.dumps(results, indent=2))
    print(f"Saved to {out}")
    final = results.get("final", results)
    print(f"pop={final.get('population', '?')} births={final.get('total_births', '?')} "
          f"gen={final.get('max_generation', '?')} div={final.get('gene_diversity', 0):.3f} "
          f"t={results.get('metadata', {}).get('elapsed_seconds', '?')}s")


async def _headless_populate(world, brain, n, rng, bear_cfg, embedder, bear_disabled, llm=None):
    """Populate world using LLM gene generation — same as live sim."""
    import random as _random
    from examples.evolutionary_ecosystem.server.gene_engine import NullBehaviorProfile, generate_npc_genes, generate_creature_name
    print(f"Creating {n} creatures via LLM (same as live sim)...")

    async def _gen_and_make(i):
        _rng = _random.Random(rng.randint(0, 2**31))
        genes = await generate_npc_genes(llm, _rng)
        name = await generate_creature_name(llm, genes, _rng)
        existing_names = {c.name for c in world.creatures.values()}
        name = _unique_name(name, existing_names, _rng)
        return await _make_creature(
            world.next_id(), genes, name,
            embedder, bear_cfg, _rng,
        )

    creatures = await asyncio.gather(*[_gen_and_make(i) for i in range(n)])
    for c in creatures:
        if bear_disabled:
            c.behavior_profile = NullBehaviorProfile()
        world.creatures[c.id] = c
        if not bear_disabled:
            r = _make_retriever(c.corpus)
            c.retriever = r
            brain.agents[c.id] = CreatureAgent(
                creature_id=c.id, retriever=r,
                llm=brain.llm,
                rng=_random.Random(rng.randint(0, 2**31)),
            )
    print(f"World ready: {len(world.creatures)} creatures")




def main():
    global args_ns
    parser = argparse.ArgumentParser(description="Evolutionary Ecosystem — BEAR Demo")
    parser.add_argument("--creatures",       type=int,   default=4)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--tick-rate",       type=int,   default=10)
    parser.add_argument("--ws-fps",          type=int,   default=10)
    parser.add_argument("--host",            default="127.0.0.1")
    parser.add_argument("--port",            type=int,   default=8003)
    parser.add_argument("--model",    default=None)
    parser.add_argument("--base-url", default=None, dest="base_url")
    parser.add_argument("--headless",       action="store_true",
                        help="Run headlessly for N ticks and write results JSON (no WebSocket server)")
    parser.add_argument("--ticks",           type=int,   default=None,
                        help="Stop after N ticks and save results (eval mode; works with or without --headless)")
    parser.add_argument("--output",          default=None,
                        help="Output JSON path when --ticks is set")
    parser.add_argument("--bear-disabled",   action="store_true", dest="bear_disabled",
                        help="Replace behavior profiles with NullBehaviorProfile (BEAR Off condition)")
    parser.add_argument("--lock-epoch",    default=None, dest="lock_epoch",
                        help="Lock simulation to a specific epoch name (e.g. predator_bloom)")
    parser.add_argument("--recombination", choices=["locus", "blend", "splice"], default="locus",
                        help="Breeding recombination method")
    parser.add_argument("--ploidy", choices=["haploid", "diploid_dominant", "diploid_codominant"],
                        default="haploid", help="Inheritance ploidy mode")
    parser.add_argument("--chunk-size", type=int, default=None, dest="chunk_size",
                        help="Rotate the output file every N ticks. Each chunk is written "
                             "as <output>_NN.<ext> (zero-padded). Keeps individual files "
                             "under GitHub's 50MB limit on long action-logged runs.")
    args_ns = parser.parse_args()

    if args_ns.headless:
        asyncio.run(_run_headless(args_ns))
    else:
        import uvicorn
        logger.info("Starting at http://%s:%d", args_ns.host, args_ns.port)
        uvicorn.run(app, host=args_ns.host, port=args_ns.port, log_level="warning")


if __name__ == "__main__":
    main()
