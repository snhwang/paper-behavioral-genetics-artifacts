"""eval10_memory_inheritance.py — validate the Lamarckian memory-inheritance channel.

Demonstrated chain:
    experience  ->  LLM-extracted memory  ->  heritable instruction  ->  expressed
    in descendants across generations.

Two phases:

  Phase 1 (--extract):  Runs the REAL bear.memory.LLMMemoryExtractor on scripted
      behavioural dialogues, using the configured LLM (default: Ollama gemma4:e2b,
      the same family used in the live simulations).  The extracted memory
      directives are cached to eval10_memories.json.  This is the only step that
      needs an LLM; it is run once.

  Phase 2 (default):  Fully headless and deterministic (no LLM).  For each cached
      memory it (a) injects the memory into a parent genome, (b) breeds N offspring
      against a fixed partner at mutation rate 0, (c) measures transmission rate,
      (d) measures expression as retrieval recall of the inherited memory for a
      topic-relevant probe query, (e) checks behaviour-profile dominance, and
      (f) breeds carriers again to measure F2 persistence.  Controls are identical
      parents WITHOUT the injected memory.  Results -> eval10_results.json.

Usage:
    # one-time extraction (needs an OpenAI-compatible LLM endpoint)
    PYTHONPATH=. python evolutionary_ecosystem/eval/eval10_memory_inheritance.py \
        --extract --base-url http://172.24.0.1:11434/v1 --model gemma4:e2b

    # deterministic measurement (no LLM)
    PYTHONPATH=. python evolutionary_ecosystem/eval/eval10_memory_inheritance.py \
        --n-genomes 6 --n-offspring 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import statistics
import time
from pathlib import Path

from bear.llm import LLM, LLMBackend
from bear.memory import LLMMemoryExtractor, ExperienceEvent
from bear.models import Context, Instruction, InstructionType, ScopeCondition

from evolutionary_ecosystem.eval.harness import (
    GENE_BANK,
    make_creature,
    breed_bear_diploid,
    make_locus_registry,
    get_embedder,
)
from evolutionary_ecosystem.server.gene_engine import GENE_CATEGORIES

_HERE = Path(__file__).resolve().parent
MEMORIES_PATH = _HERE / "eval10_memories.json"
RESULTS_PATH = _HERE / "eval10_results.json"


# ---------------------------------------------------------------------------
# Phase 1: scripted behavioural dialogues -> real LLM extraction
# ---------------------------------------------------------------------------
# Each scenario is 8 exchanges (matching the extractor's default batch_size) in
# which an agent recounts a behavioural lesson learned through experience.  The
# probe_query is a DIFFERENT, topic-relevant question used later to test whether
# the inherited memory is retrievable (expressed) in descendants.

SCENARIOS = [
    {
        "name": "predator_flee",
        "agent_name": "Scout",
        "probe_query": "A predator is approaching me right now. What should I do?",
        "turns": [
            ("What happened when the hawk came over the ridge?",
             "I learned the hard way that standing still gets you caught. The moment I see a predator now I flee immediately toward the nearest cover."),
            ("Did fighting back ever work for you?",
             "Never. Every time I tried to stand my ground against a predator I nearly died. Fleeing early is the only thing that keeps me alive."),
            ("How do you know when to run?",
             "The instant I sense a predator I do not wait to confirm it. I run first and assess from safety. Hesitation is what gets creatures killed."),
            ("Where do you run to?",
             "Always toward dense cover or a burrow. Open ground is a death trap when a predator is hunting. I memorised every hiding spot in my range."),
            ("What about your young?",
             "I teach them the same rule. When a predator appears, flee toward cover without hesitation. It is the most important survival lesson there is."),
            ("Do you ever freeze instead of running?",
             "Only as a last resort if cover is too far. But my first instinct, the one that saved me again and again, is to flee the moment danger appears."),
            ("Has this rule ever failed you?",
             "Not once. Fleeing immediately toward cover from any predator has kept me alive through three predator blooms. I trust it completely."),
            ("Any final word on predators?",
             "Treat every predator as lethal and flee at once toward the nearest cover. Speed and early reaction beat strength every single time."),
        ],
    },
    {
        "name": "forage_caution",
        "agent_name": "Forager",
        "probe_query": "I am hungry and found some food. How should I approach eating it?",
        "turns": [
            ("How did you get so good at finding food?",
             "I learned to remember every productive feeding spot and return to them in order. Wandering randomly wastes energy you cannot afford."),
            ("Do you eat the first food you find?",
             "No. I learned to observe a food source briefly before eating, because the best patches are often where predators wait in ambush."),
            ("What changed how you forage?",
             "After a near miss at a berry patch, I now always check for danger before feeding, then eat quickly and move on rather than lingering."),
            ("How do you handle competition for food?",
             "I avoid fighting over food. I learned it is better to slip in, take a quick share, and leave than to risk injury in a contest."),
            ("Do you share what you find?",
             "I lead my kin to good patches once I know they are safe. A fed group survives the lean epochs far better than a hoarder does."),
            ("What about new foods?",
             "I test anything unfamiliar with a small bite first. Caution before a full meal has saved me from sickness more than once."),
            ("How do you forage in famine?",
             "I range wider and rely on the memorised map of old patches. Knowing where food was last season is the difference between life and starvation."),
            ("Sum up your approach to food.",
             "Scout carefully, check for danger before eating, take a quick share, and remember every good patch for the hard times ahead."),
        ],
    },
    {
        "name": "stealth_hide",
        "agent_name": "Shade",
        "probe_query": "Danger is near and I want to avoid being seen. What is the best approach?",
        "turns": [
            ("Why do you move so quietly?",
             "I learned that staying unseen is safer than being fast. I keep my body low, move slowly, and freeze the instant anything notices me."),
            ("When did stealth save you?",
             "Countless times. By flattening against cover and controlling my breathing I let predators pass within a stride of me without being noticed."),
            ("Is hiding better than running?",
             "Often, yes. If I am already concealed, the best move is to hold absolutely still rather than break cover and draw attention."),
            ("How do you stay hidden so long?",
             "Patience. I learned to wait motionless far longer than feels necessary, because predators often linger after they seem to have gone."),
            ("Do you teach this to others?",
             "Yes. I show my young to match their surroundings, move silently, and freeze when watched. Concealment is a skill that must be learned early."),
            ("What gives away a hiding creature?",
             "Movement and noise. I learned to plan a silent path in advance, stepping only where I will not rustle or snap anything underfoot."),
            ("Has stealth ever failed?",
             "Only when I panicked and bolted too soon. The lesson stuck: stay concealed and still unless flight is truly the last option."),
            ("Your core rule for danger?",
             "Stay low, stay silent, and freeze when danger is near. Being unseen has kept me alive when speed alone would not have."),
        ],
    },
    {
        "name": "rally_cooperate",
        "agent_name": "Caller",
        "probe_query": "The group is in danger. How should I respond to help everyone survive?",
        "turns": [
            ("Why do you call out when danger comes?",
             "I learned that a lone warning saves the whole group. The moment I spot a predator I raise a loud alarm and rally everyone to safe ground."),
            ("Does warning others put you at risk?",
             "It does, but I learned the group survives far better when someone sounds the alarm. Collective defence beats every creature fleeing alone."),
            ("How did you discover this?",
             "After a bloom scattered my group and many died alone, I learned that rallying together and moving as one keeps far more of us alive."),
            ("What do you do after the alarm?",
             "I gather the young and the slow toward a defended rally point, because stragglers are what predators pick off first."),
            ("Do others follow you?",
             "They learned to. A clear, consistent alarm call that everyone recognises turns a panicked scatter into an organised retreat."),
            ("Is cooperation worth the cost?",
             "Always. I learned that a coordinated group withstands pressure that would destroy isolated creatures. We are stronger together."),
            ("What about food and cooperation?",
             "The same lesson holds. Sharing good patches and watching for each other lets the whole group ride out the lean epochs."),
            ("Your guiding principle?",
             "Sound the alarm early and rally the group to safety. Coordinated defence and shared effort are what carry a population through danger."),
        ],
    },
]


_CONTENT_RE = re.compile(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', re.S)


def _parse_memories_robust(raw: str, max_n: int) -> list[dict]:
    """Extract (content, topics) pairs from the model's response.

    The extraction PROMPT and the LLM are identical to bear.memory's default
    LLMMemoryExtractor; only the parsing is hardened.  Small local models
    (gemma-4-E2B) reliably quote the ``content`` string but often emit
    ``"topics": [survival, predator]`` with unquoted tokens, which breaks strict
    ``json.loads``.  We first try strict JSON, then fall back to scraping the
    quoted content strings and their following topic arrays.  This tolerates the
    malformed JSON and truncated tails without changing what is extracted.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"```[a-zA-Z]*", "", raw).replace("```", "").strip()
    try:
        v = json.loads(raw)
        if isinstance(v, list):
            return [m for m in v if isinstance(m, dict) and m.get("content")][:max_n]
    except Exception:
        pass
    mems: list[dict] = []
    for m in _CONTENT_RE.finditer(raw):
        content = m.group(1).strip()
        if not content:
            continue
        tail = raw[m.end(): m.end() + 300]
        topics: list[str] = []
        tm = re.search(r'"topics"\s*:\s*\[([^\]]*)\]', tail, re.S)
        if tm:
            for tok in tm.group(1).split(","):
                tok = tok.strip().strip('"').strip("'").strip()
                if tok and " " not in tok:
                    topics.append(tok.lower())
        mems.append({"content": content, "topics": topics[:5]})
        if len(mems) >= max_n:
            break
    return mems


class RobustMemoryExtractor(LLMMemoryExtractor):
    """bear.memory.LLMMemoryExtractor with the identical prompt/cadence but a
    parse tolerant of the small model's malformed JSON (see _parse_memories_robust)."""

    async def _extract_from_batch(self, batch, llm):
        agent_id = batch[0].agent_id
        name = self._agent_name or batch[0].metadata.get("agent_name", "") or agent_id
        conversation = "\n".join(
            f"User: {ev.query}\n{name}: {ev.response}" for ev in batch
        )
        try:
            resp = await llm.generate(
                system=(
                    f"Extract 1-{self._max_per_batch} memorable facts or opinions that "
                    f"{name} shared in this conversation. "
                    f"Output a JSON array of objects, each with:\n"
                    f'"content": one sentence (what {name} shared)\n'
                    f'"topics": list of 3-5 topic tags (lowercase single words)\n'
                    f"If nothing memorable was shared, output [].\n"
                    f"JSON only, no markdown."
                ),
                user=conversation,
                temperature=0.3,
                max_tokens=200,
            )
        except Exception:
            return []
        memories = _parse_memories_robust(resp.content, self._max_per_batch)
        instructions: list[Instruction] = []
        now = time.time()
        for mem in memories:
            content = str(mem.get("content", "")).strip()
            topics = [str(t) for t in mem.get("topics", [])][:5]
            if not content:
                continue
            self._count += 1
            instructions.append(Instruction(
                id=f"memory-{agent_id}-{int(now)}-{self._count}",
                type=InstructionType.DIRECTIVE,
                priority=self._priority,
                content=(
                    f"Memory — {name} previously shared: {content}\n"
                    f"Mention this naturally if the topic comes up again."
                ),
                scope=ScopeCondition(tags=[agent_id] + topics),
                tags=[self._memory_tag, agent_id] + topics,
                metadata={"source": "memory_extractor", "created": now},
            ))
        return instructions


async def _extract_memories(llm: LLM) -> list[dict]:
    records: list[dict] = []
    for sc in SCENARIOS:
        # batch_size 8 matches the paper's default extraction cadence.
        extractor = RobustMemoryExtractor(agent_name=sc["agent_name"], batch_size=8)
        produced: list[Instruction] = []
        for q, r in sc["turns"]:
            ev = ExperienceEvent(agent_id=sc["agent_name"].lower(), query=q, response=r)
            produced.extend(await extractor.process(ev, llm))
        print(f"  scenario {sc['name']:16s} -> {len(produced)} memory instruction(s)")
        for inst in produced:
            records.append({
                "scenario":    sc["name"],
                "probe_query": sc["probe_query"],
                "id":          inst.id,
                "priority":    inst.priority,
                "content":     inst.content,
                "scope_tags":  list(inst.scope.tags),
                "tags":        list(inst.tags),
                "metadata":    dict(inst.metadata),
            })
    return records


def run_extract(args) -> None:
    print(f"[extract] LLM backend=openai-compat base_url={args.base_url} model={args.model}")
    llm = LLM(backend=LLMBackend.OPENAI, model=args.model, base_url=args.base_url)
    records = asyncio.run(_extract_memories(llm))
    MEMORIES_PATH.write_text(json.dumps(records, indent=2))
    print(f"[extract] wrote {len(records)} memories -> {MEMORIES_PATH}")


# ---------------------------------------------------------------------------
# Phase 2: headless, deterministic inheritance measurement
# ---------------------------------------------------------------------------

def _memory_from_record(rec: dict) -> Instruction:
    return Instruction(
        id=rec["id"],
        type=InstructionType.DIRECTIVE,
        priority=rec["priority"],
        content=rec["content"],
        scope=ScopeCondition(tags=rec["scope_tags"]),
        tags=rec["tags"],
        metadata=rec["metadata"],
    )


def _memory_instances(corpus) -> list[Instruction]:
    """All instructions in a corpus that descend from an injected memory."""
    out = []
    for inst in corpus.filter():
        meta = inst.metadata or {}
        if (
            "memory" in (inst.tags or [])
            or meta.get("source") == "memory_extractor"
            or str(inst.id).startswith("memory-")
            or str(meta.get("original_id", "")).startswith("memory-")
        ):
            out.append(inst)
    return out


def _recalls_memory(retriever, probe_query: str, topics: list[str], top_k: int = 5) -> bool:
    """True if a memory-derived instruction is retrieved for the probe query."""
    ctx = Context(tags=topics)
    scored = retriever.retrieve(query=probe_query, context=ctx, top_k=top_k, threshold=0.0)
    for s in scored:
        meta = s.instruction.metadata or {}
        if (
            "memory" in (s.instruction.tags or [])
            or meta.get("source") == "memory_extractor"
            or str(meta.get("original_id", "")).startswith("memory-")
        ):
            return True
    return False


def _profile_dominated_by_memory(creature) -> int:
    """How many of the behaviour-profile situations a memory wins as top-1."""
    bp = creature.behavior_profile
    if bp is None:
        return 0
    n = 0
    for r in bp.situations.values():
        if r.gene_text and r.gene_text.strip().startswith("Memory"):
            n += 1
    return n


def run_measure(args) -> None:
    if not MEMORIES_PATH.exists():
        raise SystemExit(f"No cached memories at {MEMORIES_PATH}. Run with --extract first.")
    memories = json.loads(MEMORIES_PATH.read_text())
    print(f"[measure] loaded {len(memories)} cached memories")

    embedder = get_embedder()  # warm the shared embedder
    registry = make_locus_registry()  # haploid, locus-based

    per_memory = []
    for mi, rec in enumerate(memories):
        rng = random.Random(args.seed + mi)
        mem_topics = rec["tags"]
        n_carriers = 0
        n_offspring = 0
        recall_carrier = 0
        recall_control_pos = 0
        dom_memory = 0
        dom_control = 0
        f2_total = 0
        f2_carriers = 0
        f2_recall = 0

        for gi in range(args.n_genomes):
            base = GENE_BANK[gi % len(GENE_BANK)]
            partner_genes = GENE_BANK[(gi + 3) % len(GENE_BANK)]
            base_genes = {c: base.get(c, "") for c in GENE_CATEGORIES}

            # --- memory-condition parent: inject the extracted memory ---
            pa = make_creature(f"pa_{mi}_{gi}", base_genes, f"PA_{mi}_{gi}", rng)
            mem = _memory_from_record(rec)
            # give the memory a unique id per parent so detection is unambiguous
            mem.id = f"memory-inj-{mi}-{gi}"
            mem.metadata = {**mem.metadata, "source": "memory_extractor"}
            pa.corpus.add(mem)
            pa.retriever.build_index()

            # --- control parent: identical genome, no memory ---
            pc = make_creature(f"pc_{mi}_{gi}", base_genes, f"PC_{mi}_{gi}", rng)
            partner = make_creature(f"pb_{mi}_{gi}",
                                    {c: partner_genes.get(c, "") for c in GENE_CATEGORIES},
                                    f"PB_{mi}_{gi}", rng)

            for oi in range(args.n_offspring):
                # memory condition
                child = breed_bear_diploid(pa, partner, f"ch_{mi}_{gi}_{oi}",
                                           f"CH_{mi}_{gi}_{oi}", rng,
                                           registry=registry, mutation_rate=0.0)
                n_offspring += 1
                carriers = _memory_instances(child.corpus)
                if carriers:
                    n_carriers += 1
                    if _recalls_memory(child.retriever, rec["probe_query"], mem_topics):
                        recall_carrier += 1
                    dom_memory += _profile_dominated_by_memory(child)

                    # --- F2: breed a carrier against a fresh partner ---
                    f2_partner = make_creature(
                        f"f2pb_{mi}_{gi}_{oi}",
                        {c: GENE_BANK[(gi + 5) % len(GENE_BANK)].get(c, "") for c in GENE_CATEGORIES},
                        f"F2PB_{mi}_{gi}_{oi}", rng)
                    grand = breed_bear_diploid(child, f2_partner,
                                               f"gc_{mi}_{gi}_{oi}", f"GC_{mi}_{gi}_{oi}",
                                               rng, registry=registry, mutation_rate=0.0)
                    f2_total += 1
                    if _memory_instances(grand.corpus):
                        f2_carriers += 1
                        if _recalls_memory(grand.retriever, rec["probe_query"], mem_topics):
                            f2_recall += 1

                # control condition
                cchild = breed_bear_diploid(pc, partner, f"cc_{mi}_{gi}_{oi}",
                                            f"CC_{mi}_{gi}_{oi}", rng,
                                            registry=registry, mutation_rate=0.0)
                if _memory_instances(cchild.corpus):
                    recall_control_pos += 1  # should be ~0: no memory injected
                dom_control += _profile_dominated_by_memory(cchild)

        rate = n_carriers / n_offspring if n_offspring else 0.0
        rec_rate = recall_carrier / n_carriers if n_carriers else 0.0
        f2_rate = f2_carriers / f2_total if f2_total else 0.0
        f2_rec_rate = f2_recall / f2_carriers if f2_carriers else 0.0
        per_memory.append({
            "scenario": rec["scenario"],
            "offspring": n_offspring,
            "transmission_rate": round(rate, 4),
            "carriers": n_carriers,
            "recall_carrier_count": recall_carrier,
            "expression_recall_in_carriers": round(rec_rate, 4),
            "control_false_carriers": recall_control_pos,
            "situations_dominated_memory": dom_memory,
            "situations_dominated_control": dom_control,
            "f2_total": f2_total,
            "f2_carriers": f2_carriers,
            "f2_recall": f2_recall,
            "f2_transmission_rate": round(f2_rate, 4),
            "f2_expression_recall": round(f2_rec_rate, 4),
        })
        print(f"  [{rec['scenario']:16s}] transmit={rate:.3f} "
              f"recall={rec_rate:.3f} f2_transmit={f2_rate:.3f} "
              f"ctrl_false={recall_control_pos} dom_mem={dom_memory}")

    # pooled aggregation across all memories/offspring (tight CIs)
    tot_off = sum(m["offspring"] for m in per_memory)
    tot_carriers = sum(m["carriers"] for m in per_memory)
    tot_recall = sum(m["recall_carrier_count"] for m in per_memory)
    tot_f2 = sum(m["f2_total"] for m in per_memory)
    tot_f2_carriers = sum(m["f2_carriers"] for m in per_memory)
    tot_f2_recall = sum(m["f2_recall"] for m in per_memory)
    tot_ctrl_false = sum(m["control_false_carriers"] for m in per_memory)

    def _wilson(k, n, z=1.96):
        if n == 0:
            return (0.0, 0.0)
        p = k / n
        d = 1 + z * z / n
        c = p + z * z / (2 * n)
        h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
        return (round((c - h) / d, 4), round((c + h) / d, 4))

    summary = {
        "n_memories": len(per_memory),
        "total_offspring": tot_off,
        "transmission_rate": round(tot_carriers / tot_off, 4) if tot_off else 0.0,
        "transmission_ci95": _wilson(tot_carriers, tot_off),
        "expression_recall_in_carriers": round(tot_recall / tot_carriers, 4) if tot_carriers else 0.0,
        "expression_recall_ci95": _wilson(tot_recall, tot_carriers),
        "f2_total": tot_f2,
        "f2_transmission_rate": round(tot_f2_carriers / tot_f2, 4) if tot_f2 else 0.0,
        "f2_transmission_ci95": _wilson(tot_f2_carriers, tot_f2),
        "f2_expression_recall": round(tot_f2_recall / tot_f2_carriers, 4) if tot_f2_carriers else 0.0,
        "total_control_false_carriers": tot_ctrl_false,
        "expected_crossover_rate": 0.5,
        "config": {"n_genomes": args.n_genomes, "n_offspring": args.n_offspring,
                   "seed": args.seed, "mutation_rate": 0.0, "crossover_rate": 0.5},
        "per_memory": per_memory,
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    print("\n[measure] SUMMARY (pooled)")
    print(f"  offspring bred          : {tot_off}")
    print(f"  transmission rate       : {summary['transmission_rate']:.3f} "
          f"CI95={summary['transmission_ci95']} (expected 0.5)")
    print(f"  expression recall       : {summary['expression_recall_in_carriers']:.3f} "
          f"CI95={summary['expression_recall_ci95']} ({tot_carriers} carriers)")
    print(f"  F2 transmission rate    : {summary['f2_transmission_rate']:.3f} "
          f"CI95={summary['f2_transmission_ci95']} ({tot_f2} F2)")
    print(f"  F2 expression recall    : {summary['f2_expression_recall']:.3f}")
    print(f"  control false carriers  : {tot_ctrl_false}")
    print(f"[measure] wrote {RESULTS_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extract", action="store_true",
                    help="run the one-time LLM extraction phase")
    ap.add_argument("--base-url", default="http://172.24.0.1:11434/v1",
                    help="OpenAI-compatible LLM endpoint for extraction")
    ap.add_argument("--model", default="gemma4:e2b")
    ap.add_argument("--n-genomes", type=int, default=6)
    ap.add_argument("--n-offspring", type=int, default=20)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    if args.extract:
        run_extract(args)
    else:
        run_measure(args)


if __name__ == "__main__":
    main()
