#!/usr/bin/env python3
"""Evaluation 6b: Genome-conditioned LLM output diversity.

Extends eval6 (retrieval-level measurement) by actually running an LLM to
validate that different creature genomes produce measurably different LLM
outputs when BEAR-retrieved guidance is used as system prompt conditioning.

Design:
  - Same 4 creatures (Bold, Timid, Curious, Calm) and 5 scenarios as eval6
  - For each creature × scenario, retrieve top-3 guidance via BEAR, compose
    into a system prompt, and call the LLM with the scenario query
  - Measure inter-creature LLM output divergence using 3 metrics:
    - Content diff ratio (token-level multiset symmetric difference)
    - Semantic cosine distance (embedding distance between full responses)
    - Hausdorff distance (sentence-level max nearest-neighbor distance)
  - Compare BEAR-conditioned divergence against a static baseline (identical
    generic prompt for all creatures → divergence ≈ 0)
  - Two temperature conditions: 0.0 (clean causal signal) and 0.7 (realistic)

Models:
- Embedding: BAAI/bge-base-en-v1.5 (768-dim sentence-transformer)
- LLM: mlx-community/mistral-nemo-instruct-2407:3 (default, configurable)

Parameters:
- Temperature: 0.0 and 0.7
- max_tokens: 200
- 4 creatures × 5 scenarios × 2 conditions (BEAR + static) × 2 temps = 160 calls

Outputs:
- eval6b_results.json  — Per-creature/scenario LLM responses and divergence metrics
- eval6b_llm_dialogue.png  — Divergence comparison charts

Usage:
    python eval6b_llm_dialogue.py                          # auto-detect backend
    python eval6b_llm_dialogue.py --backend local          # LM Studio
    python eval6b_llm_dialogue.py --backend anthropic      # Claude Haiku
    python eval6b_llm_dialogue.py --model MODEL_ID         # specific model
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from bear import Config, EmbeddingBackend
from bear.retriever import Retriever, Embedder
from bear.models import Context

from evolutionary_ecosystem.eval.harness import (
    GENE_BANK,
    SITUATION_NAMES,
    cosine_similarity,
    ensure_eval_patched,
    get_config,
    get_embedder,
    profile_to_vector,
)
from evolutionary_ecosystem.server.gene_engine import (
    build_corpus,
    compute_behavior_profile,
)

OUT_DIR = Path(__file__).resolve().parents[2] / "results"

# ---------------------------------------------------------------------------
# LLM backend detection and calling (follows eval_output_divergence.py pattern)
# ---------------------------------------------------------------------------

DEFAULT_LOCAL_MODEL = "mistral-nemo-instruct-2407"
try:
    from bear.utils import detect_local_llm_url
    DEFAULT_LOCAL_URL = detect_local_llm_url()
except ImportError:
    DEFAULT_LOCAL_URL = "http://127.0.0.1:1234/v1"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


def _call_anthropic(system_prompt: str, user_message: str,
                    model: str, temperature: float) -> str:
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=200,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text.strip()


def _call_local(system_prompt: str, user_message: str,
                model: str, base_url: str, temperature: float) -> str:
    import urllib.request
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": 200,
    }, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


def make_call_llm(backend: str, model: str, base_url: str):
    """Return a call_llm(system, user, temperature) closure."""
    if backend == "anthropic":
        def call_llm(system_prompt: str, user_message: str,
                     temperature: float = 0.0) -> str:
            try:
                return _call_anthropic(system_prompt, user_message,
                                       model, temperature)
            except Exception as e:
                print(f"\n  LLM error: {e}")
                return f"[ERROR: {e}]"
    else:
        def call_llm(system_prompt: str, user_message: str,
                     temperature: float = 0.0) -> str:
            try:
                return _call_local(system_prompt, user_message,
                                   model, base_url, temperature)
            except Exception as e:
                print(f"\n  LLM error: {e}")
                return f"[ERROR: {e}]"
    return call_llm


def detect_backend(args):
    """Auto-detect backend, model, and base_url from args + environment."""
    if args.backend == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ERROR: --backend anthropic requires ANTHROPIC_API_KEY")
            sys.exit(1)
        model = args.model or DEFAULT_ANTHROPIC_MODEL
        return "anthropic", model, ""

    # Any explicit --base-url routes to the OpenAI-compatible local backend
    # (works for Ollama, LM Studio, or any compatible server)
    if args.backend == "local" or (args.backend == "auto" and getattr(args, "base_url", None) and args.base_url != DEFAULT_LOCAL_URL):
        model = args.model or DEFAULT_LOCAL_MODEL
        return "local", model, args.base_url


    # Auto-detect
    if os.environ.get("ANTHROPIC_API_KEY"):
        model = args.model or DEFAULT_ANTHROPIC_MODEL
        return "anthropic", model, ""

    model = args.model or DEFAULT_LOCAL_MODEL
    try:
        import urllib.request
        urllib.request.urlopen(f"{args.base_url}/models", timeout=3)
        return "local", model, args.base_url
    except Exception:
        pass

    print("ERROR: No LLM backend available.")
    print("  Option 1: export ANTHROPIC_API_KEY=sk-...")
    print(f"  Option 2: start a local LLM server at {args.base_url}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Divergence metrics (from eval_output_divergence.py)
# ---------------------------------------------------------------------------

def content_diff_ratio(text_a: str, text_b: str) -> float:
    """Fraction of tokens that differ (multiset symmetric difference)."""
    tokens_a = text_a.split()
    tokens_b = text_b.split()
    if not tokens_a and not tokens_b:
        return 0.0
    count_a = Counter(tokens_a)
    count_b = Counter(tokens_b)
    all_tokens = set(count_a.keys()) | set(count_b.keys())
    diff_count = sum(abs(count_a.get(t, 0) - count_b.get(t, 0))
                     for t in all_tokens)
    total_count = len(tokens_a) + len(tokens_b)
    return diff_count / total_count if total_count > 0 else 0.0


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p]


def hausdorff_on_sentences(embedder: Embedder, text_a: str,
                           text_b: str) -> float:
    sents_a = split_sentences(text_a)
    sents_b = split_sentences(text_b)
    if not sents_a or not sents_b:
        return 1.0
    embs_a = [embedder.embed_single(s) for s in sents_a]
    embs_b = [embedder.embed_single(s) for s in sents_b]

    def directed(src, tgt):
        return max(min(cosine_distance(s, t) for t in tgt) for s in src)

    return max(directed(embs_a, embs_b), directed(embs_b, embs_a))


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def compute_statistical_tests(bear_values: list[float],
                              static_values: list[float],
                              metric_name: str) -> dict:
    """Compute paired statistical tests comparing BEAR vs static divergence.

    Each element in bear_values / static_values is one paired observation
    (same creature-pair × scenario measured under both conditions).
    """
    bear_arr = np.array(bear_values)
    static_arr = np.array(static_values)
    diffs = bear_arr - static_arr
    n = len(diffs)

    result: dict = {"metric": metric_name, "n_pairs": n}

    # --- Means and 95% confidence intervals ---
    for label, arr in [("bear", bear_arr), ("static", static_arr)]:
        mean = float(np.mean(arr))
        se = float(np.std(arr, ddof=1) / np.sqrt(n))
        t_crit = float(scipy_stats.t.ppf(0.975, df=n - 1))
        result[f"{label}_mean"] = round(mean, 6)
        result[f"{label}_std"] = round(float(np.std(arr, ddof=1)), 6)
        result[f"{label}_ci95_lo"] = round(mean - t_crit * se, 6)
        result[f"{label}_ci95_hi"] = round(mean + t_crit * se, 6)

    # --- Difference stats ---
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    result["mean_diff"] = round(mean_diff, 6)
    result["std_diff"] = round(std_diff, 6)

    # --- Cohen's d (paired) ---
    cohens_d = mean_diff / std_diff if std_diff > 0 else float("inf")
    result["cohens_d"] = round(cohens_d, 4)

    # --- Shapiro-Wilk on differences (normality check) ---
    if n >= 3:
        sw_stat, sw_p = scipy_stats.shapiro(diffs)
        result["shapiro_wilk_stat"] = round(float(sw_stat), 6)
        result["shapiro_wilk_p"] = round(float(sw_p), 6)
    else:
        result["shapiro_wilk_stat"] = None
        result["shapiro_wilk_p"] = None

    # --- Paired t-test ---
    t_stat, t_p = scipy_stats.ttest_rel(bear_arr, static_arr)
    result["paired_ttest_t"] = round(float(t_stat), 6)
    result["paired_ttest_p"] = round(float(t_p), 6)

    # --- Wilcoxon signed-rank test (non-parametric) ---
    # Need at least one non-zero difference
    if np.any(diffs != 0):
        try:
            w_stat, w_p = scipy_stats.wilcoxon(diffs)
            result["wilcoxon_stat"] = round(float(w_stat), 6)
            result["wilcoxon_p"] = round(float(w_p), 6)
        except ValueError:
            # Too few samples for Wilcoxon
            result["wilcoxon_stat"] = None
            result["wilcoxon_p"] = None
    else:
        result["wilcoxon_stat"] = None
        result["wilcoxon_p"] = None

    return result


# ---------------------------------------------------------------------------
# Scenarios and creatures (same as eval6)
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "name": "idle_exploration",
        "query": "I'm feeling content with happiness 72/100. What should I do? "
                 "[Epoch: abundance, Weather: clear]",
        "tags": ["idle", "mood_happy", "happy", "epoch:abundance",
                 "weather:clear"],
    },
    {
        "name": "predator_attack",
        "query": "A predator is attacking! (predator at dist 4.2) Do I flee "
                 "or rally others to defend? [Epoch: predator_bloom, "
                 "Weather: storm]",
        "tags": ["predator", "danger", "flee", "epoch:predator_bloom",
                 "weather:storm"],
    },
    {
        "name": "social_greeting",
        "query": "I'm meeting Omega for the first time (distance 2.3). How "
                 "do I react given my personality? [Epoch: abundance, "
                 "Weather: clear]",
        "tags": ["greeting", "social", "epoch:abundance", "weather:clear"],
    },
    {
        "name": "hungry_survival",
        "query": "I'm feeling cautious with happiness 35/100. Energy: 22/100. "
                 "What should I do? [Epoch: famine, Weather: rain]",
        "tags": ["idle", "mood_cautious", "unhappy", "hunger", "survival",
                 "epoch:famine", "weather:rain"],
    },
    {
        "name": "enraged_combat",
        "query": "I am ENRAGED (rage=92/100). I must challenge the nearest "
                 "creature. What do I do? [Epoch: ice_age, Weather: snow]",
        "tags": ["enraged", "aggression", "nearby", "epoch:ice_age",
                 "weather:snow"],
    },
]

CREATURE_INDICES = [0, 1, 2, 3]  # Bold, Timid, Curious, Calm
CREATURE_LABELS = ["Bold", "Timid", "Curious", "Calm"]

STATIC_SYSTEM_PROMPT = (
    "You are a creature in an evolutionary ecosystem. "
    "Respond in character. Describe what you do in 2-3 sentences."
)

BEAR_SYSTEM_TEMPLATE = (
    "You are {name}, a creature in an evolutionary ecosystem.\n"
    "Your behavioral guidance:\n{guidance}\n\n"
    "Respond in character. Describe what you do in 2-3 sentences."
)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_temperature_condition(
    creature_data: list[dict],
    call_llm,
    embedder: Embedder,
    temperature: float,
) -> dict:
    """Run all creatures × scenarios at a given temperature."""
    scenario_names = [s["name"] for s in SCENARIOS]
    n_creatures = len(creature_data)

    # Collect LLM responses: BEAR-conditioned and static
    bear_responses: dict[str, dict[str, str]] = {}  # creature -> scenario -> response
    static_responses: dict[str, dict[str, str]] = {}

    for cd in creature_data:
        label = cd["label"]
        bear_responses[label] = {}
        static_responses[label] = {}

        for scenario in SCENARIOS:
            sname = scenario["name"]

            # BEAR-conditioned prompt
            system_prompt = BEAR_SYSTEM_TEMPLATE.format(
                name=label,
                guidance=cd["guidances"][sname],
            )
            print(f"  {label} × {sname} (temp={temperature})...",
                  end=" ", flush=True)
            t0 = time.time()
            bear_resp = call_llm(system_prompt, scenario["query"], temperature)
            elapsed = time.time() - t0
            bear_responses[label][sname] = bear_resp
            print(f"({elapsed:.1f}s) {bear_resp[:60]}...")

            # Static baseline
            static_resp = call_llm(STATIC_SYSTEM_PROMPT,
                                   scenario["query"], temperature)
            static_responses[label][sname] = static_resp

    # Compute pairwise divergence metrics per scenario
    per_scenario = {}
    bear_cd_all, bear_cos_all, bear_hd_all = [], [], []
    static_cd_all, static_cos_all, static_hd_all = [], [], []

    for sname in scenario_names:
        pairs = []
        for i in range(n_creatures):
            for j in range(i + 1, n_creatures):
                la = creature_data[i]["label"]
                lb = creature_data[j]["label"]

                # BEAR divergence
                ra = bear_responses[la][sname]
                rb = bear_responses[lb][sname]
                b_cd = content_diff_ratio(ra, rb)
                emb_a = embedder.embed_single(ra)
                emb_b = embedder.embed_single(rb)
                b_cos = cosine_distance(emb_a, emb_b)
                b_hd = hausdorff_on_sentences(embedder, ra, rb)

                # Static divergence
                sa = static_responses[la][sname]
                sb = static_responses[lb][sname]
                s_cd = content_diff_ratio(sa, sb)
                emb_sa = embedder.embed_single(sa)
                emb_sb = embedder.embed_single(sb)
                s_cos = cosine_distance(emb_sa, emb_sb)
                s_hd = hausdorff_on_sentences(embedder, sa, sb)

                pairs.append({
                    "pair": f"{la}-{lb}",
                    "bear": {"content_diff": round(b_cd, 4),
                             "cosine_dist": round(b_cos, 4),
                             "hausdorff": round(b_hd, 4)},
                    "static": {"content_diff": round(s_cd, 4),
                               "cosine_dist": round(s_cos, 4),
                               "hausdorff": round(s_hd, 4)},
                })

                bear_cd_all.append(b_cd)
                bear_cos_all.append(b_cos)
                bear_hd_all.append(b_hd)
                static_cd_all.append(s_cd)
                static_cos_all.append(s_cos)
                static_hd_all.append(s_hd)

        per_scenario[sname] = {
            "pairs": pairs,
            "bear_mean": {
                "content_diff": round(float(np.mean([p["bear"]["content_diff"] for p in pairs])), 4),
                "cosine_dist": round(float(np.mean([p["bear"]["cosine_dist"] for p in pairs])), 4),
                "hausdorff": round(float(np.mean([p["bear"]["hausdorff"] for p in pairs])), 4),
            },
            "static_mean": {
                "content_diff": round(float(np.mean([p["static"]["content_diff"] for p in pairs])), 4),
                "cosine_dist": round(float(np.mean([p["static"]["cosine_dist"] for p in pairs])), 4),
                "hausdorff": round(float(np.mean([p["static"]["hausdorff"] for p in pairs])), 4),
            },
        }

    summary = {
        "bear_mean_content_diff": round(float(np.mean(bear_cd_all)), 4),
        "bear_mean_cosine_dist": round(float(np.mean(bear_cos_all)), 4),
        "bear_mean_hausdorff": round(float(np.mean(bear_hd_all)), 4),
        "static_mean_content_diff": round(float(np.mean(static_cd_all)), 4),
        "static_mean_cosine_dist": round(float(np.mean(static_cos_all)), 4),
        "static_mean_hausdorff": round(float(np.mean(static_hd_all)), 4),
        "delta_content_diff": round(float(np.mean(bear_cd_all)) - float(np.mean(static_cd_all)), 4),
        "delta_cosine_dist": round(float(np.mean(bear_cos_all)) - float(np.mean(static_cos_all)), 4),
        "delta_hausdorff": round(float(np.mean(bear_hd_all)) - float(np.mean(static_hd_all)), 4),
    }

    # --- Statistical tests (paired: BEAR vs static per creature-pair × scenario) ---
    statistical_tests = {
        "content_diff": compute_statistical_tests(
            bear_cd_all, static_cd_all, "content_diff"),
        "cosine_dist": compute_statistical_tests(
            bear_cos_all, static_cos_all, "cosine_dist"),
        "hausdorff": compute_statistical_tests(
            bear_hd_all, static_hd_all, "hausdorff"),
    }

    # Print stat summary
    print(f"\n  Statistical tests (temp={temperature}):")
    for metric, st in statistical_tests.items():
        sig_t = "***" if st["paired_ttest_p"] < 0.001 else (
                "**" if st["paired_ttest_p"] < 0.01 else (
                "*" if st["paired_ttest_p"] < 0.05 else "n.s."))
        sig_w = ""
        if st["wilcoxon_p"] is not None:
            sig_w = "***" if st["wilcoxon_p"] < 0.001 else (
                    "**" if st["wilcoxon_p"] < 0.01 else (
                    "*" if st["wilcoxon_p"] < 0.05 else "n.s."))
        else:
            sig_w = "N/A"
        print(f"    {metric}: d={st['cohens_d']:.3f}, "
              f"t={st['paired_ttest_t']:.3f} p={st['paired_ttest_p']:.4f} ({sig_t}), "
              f"Wilcoxon p={st.get('wilcoxon_p', 'N/A')} ({sig_w}), "
              f"Shapiro p={st.get('shapiro_wilk_p', 'N/A')}")
        print(f"      BEAR: {st['bear_mean']:.4f} [{st['bear_ci95_lo']:.4f}, {st['bear_ci95_hi']:.4f}]  "
              f"Static: {st['static_mean']:.4f} [{st['static_ci95_lo']:.4f}, {st['static_ci95_hi']:.4f}]")

    return {
        "temperature": temperature,
        "summary": summary,
        "statistical_tests": statistical_tests,
        "per_scenario": per_scenario,
        "bear_responses": bear_responses,
        "static_responses": static_responses,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Eval 6b: Genome-conditioned LLM output diversity")
    parser.add_argument("--model", default="",
                        help="LLM model ID (auto-detected if omitted)")
    parser.add_argument("--backend", choices=["auto", "anthropic", "local"],
                        default="auto",
                        help="LLM backend: anthropic, local, or auto. Use --base-url to point at any OpenAI-compatible endpoint (Ollama, etc.)")
    parser.add_argument("--base-url", default=DEFAULT_LOCAL_URL,
                        help=f"Local LLM server URL (default: {DEFAULT_LOCAL_URL})")
    args = parser.parse_args()

    backend, model, base_url = detect_backend(args)
    call_llm = make_call_llm(backend, model, base_url)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_eval_patched()

    embedder = get_embedder()
    config = get_config()

    print("=" * 70)
    print("EVAL 6b: Genome-Conditioned LLM Output Diversity")
    print("=" * 70)
    print(f"LLM backend: {backend}")
    print(f"LLM model: {model}")
    print(f"Base URL: {base_url}")
    print(f"Temperatures: 0.0, 0.7")
    print(f"Platform: {platform.platform()}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Build corpus + retriever for each creature, collect guidance
    creature_data = []
    for idx, label in zip(CREATURE_INDICES, CREATURE_LABELS):
        genes = GENE_BANK[idx]
        corpus = build_corpus(label, genes)
        profile = compute_behavior_profile(corpus, config,
                                           shared_embedder=embedder)
        retriever = Retriever(corpus=corpus, config=config)
        retriever.build_index()

        # Retrieve guidance for each scenario
        guidances = {}
        for scenario in SCENARIOS:
            ctx = Context(tags=scenario["tags"],
                          domain="evolutionary_ecosystem")
            scored = retriever.retrieve(
                query=scenario["query"], context=ctx, top_k=3, threshold=0.3)
            if scored:
                full_guidance = " | ".join(
                    s.instruction.content[:100] for s in scored[:3])
            else:
                full_guidance = "(no retrieval)"
            guidances[scenario["name"]] = full_guidance

        creature_data.append({
            "label": label,
            "genes": genes,
            "profile": profile_to_vector(profile),
            "corpus_size": len(corpus.instructions),
            "guidances": guidances,
        })
        print(f"{label}: corpus={len(corpus.instructions)} instructions")

    print()

    # Run both temperature conditions
    results_by_temp = {}
    for temp in [0.0, 0.7]:
        print(f"\n{'='*60}")
        print(f"Temperature = {temp}")
        print(f"{'='*60}")
        results_by_temp[str(temp)] = run_temperature_condition(
            creature_data, call_llm, embedder, temp)

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'':30s} {'Content Diff':>12s} {'Cosine Dist':>12s} {'Hausdorff':>12s}")
    for temp_key, res in results_by_temp.items():
        s = res["summary"]
        print(f"{'BEAR (temp=' + temp_key + ')':30s} "
              f"{s['bear_mean_content_diff']:12.4f} "
              f"{s['bear_mean_cosine_dist']:12.4f} "
              f"{s['bear_mean_hausdorff']:12.4f}")
        print(f"{'Static (temp=' + temp_key + ')':30s} "
              f"{s['static_mean_content_diff']:12.4f} "
              f"{s['static_mean_cosine_dist']:12.4f} "
              f"{s['static_mean_hausdorff']:12.4f}")
        print(f"{'Delta (temp=' + temp_key + ')':30s} "
              f"{s['delta_content_diff']:+12.4f} "
              f"{s['delta_cosine_dist']:+12.4f} "
              f"{s['delta_hausdorff']:+12.4f}")
        print()

    # LaTeX table
    print("\n% === LaTeX Table ===")
    print("\\begin{table}[t]")
    print("\\caption{Genome-conditioned LLM output diversity: inter-creature")
    print("response divergence when BEAR-retrieved guidance conditions the")
    print("system prompt vs.\\ a static generic prompt.}")
    print("\\label{tab:genome-llm-divergence}")
    print("\\begin{tabular}{@{}llccc@{}}")
    print("\\toprule")
    print("Temp & Condition & Content Diff & Cosine Dist & Hausdorff \\\\")
    print("\\midrule")
    for temp_key, res in results_by_temp.items():
        s = res["summary"]
        print(f"{temp_key} & BEAR & {s['bear_mean_content_diff']:.3f} & "
              f"{s['bear_mean_cosine_dist']:.3f} & "
              f"{s['bear_mean_hausdorff']:.3f} \\\\")
        print(f" & Static & {s['static_mean_content_diff']:.3f} & "
              f"{s['static_mean_cosine_dist']:.3f} & "
              f"{s['static_mean_hausdorff']:.3f} \\\\")
        print(f" & $\\Delta$ & {s['delta_content_diff']:+.3f} & "
              f"{s['delta_cosine_dist']:+.3f} & "
              f"{s['delta_hausdorff']:+.3f} \\\\")
        print("\\midrule")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    # Save results
    output = {
        "metadata": {
            "eval": "6b",
            "description": "Genome-conditioned LLM output diversity",
            "llm_backend": backend,
            "llm_model": model,
            "base_url": base_url,
            "temperatures": [0.0, 0.7],
            "platform": platform.platform(),
            "timestamp": datetime.now().isoformat(),
            "n_creatures": len(CREATURE_INDICES),
            "n_scenarios": len(SCENARIOS),
            "creature_labels": CREATURE_LABELS,
            "scenario_names": [s["name"] for s in SCENARIOS],
        },
        "per_creature": {
            cd["label"]: {
                "genes": cd["genes"],
                "behavior_profile": {s: round(cd["profile"][i], 4)
                                     for i, s in enumerate(SITUATION_NAMES)},
                "corpus_size": cd["corpus_size"],
                "guidances": cd["guidances"],
            }
            for cd in creature_data
        },
        "results_by_temperature": {
            temp_key: {
                "summary": res["summary"],
                "statistical_tests": res["statistical_tests"],
                "per_scenario": res["per_scenario"],
                "bear_responses": res["bear_responses"],
                "static_responses": res["static_responses"],
            }
            for temp_key, res in results_by_temp.items()
        },
    }

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    results_path = OUT_DIR / "eval6b_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, cls=_NumpyEncoder)
    print(f"\nResults saved to {results_path}")

    try:
        _plot_divergence(output)
    except Exception as e:
        print(f"Chart generation failed: {e}")


def _plot_divergence(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    scenario_names = output["metadata"]["scenario_names"]
    creature_labels = output["metadata"]["creature_labels"]
    n_creatures = len(creature_labels)

    # Readable scenario labels
    scenario_display = {
        "idle_exploration": "Idle\nExploration",
        "predator_attack": "Predator\nAttack",
        "social_greeting": "Social\nGreeting",
        "hungry_survival": "Hungry\nSurvival",
        "enraged_combat": "Enraged\nCombat",
    }

    # Color palette
    C_BEAR = "#1565C0"     # deep blue
    C_STATIC = "#E65100"   # deep orange
    C_BEAR_L = "#42A5F5"   # light blue (temp=0.7)
    C_STATIC_L = "#FF9800" # light orange (temp=0.7)

    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.30,
                  left=0.08, right=0.96, top=0.91, bottom=0.06)

    # ─── Panel A: All 3 metrics, both temps (main result) ───
    ax_a = fig.add_subplot(gs[0, :])
    metrics = ["content_diff", "cosine_dist", "hausdorff"]
    metric_labels = ["Content Diff", "Cosine Distance", "Hausdorff"]
    temps = ["0.0", "0.7"]

    x = np.arange(len(metrics))
    total_width = 0.7
    bar_w = total_width / 4  # 4 bars per metric group
    offsets = [-1.5, -0.5, 0.5, 1.5]
    bar_colors = [C_BEAR, C_STATIC, C_BEAR_L, C_STATIC_L]
    bar_labels = [
        "BEAR (temp=0.0)", "Static (temp=0.0)",
        "BEAR (temp=0.7)", "Static (temp=0.7)",
    ]
    bar_hatches = [None, None, "//", "//"]

    for bi, (label, color, hatch) in enumerate(
            zip(bar_labels, bar_colors, bar_hatches)):
        t_idx = bi // 2      # 0 for temp=0.0, 1 for temp=0.7
        is_static = bi % 2   # 0 for BEAR, 1 for static
        t_key = temps[t_idx]
        prefix = "static" if is_static else "bear"
        res = output["results_by_temperature"][t_key]["summary"]
        vals = [res[f"{prefix}_mean_{m}"] for m in metrics]
        bars = ax_a.bar(x + offsets[bi] * bar_w, vals, bar_w,
                        label=label, color=color, alpha=0.85,
                        hatch=hatch, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if abs(v) > 0.005:
                ax_a.text(bar.get_x() + bar.get_width() / 2,
                          bar.get_height() + 0.012,
                          f"{v:.3f}", ha="center", va="bottom",
                          fontsize=8, fontweight="bold")
            else:
                ax_a.text(bar.get_x() + bar.get_width() / 2, 0.015,
                          "0.000", ha="center", va="bottom",
                          fontsize=7, color="gray")

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(metric_labels, fontsize=11)
    ax_a.set_ylabel("Mean Inter-Creature Divergence", fontsize=11)
    ax_a.set_title("(a)  BEAR-Conditioned vs Static Baseline: "
                    "Three Divergence Metrics",
                    fontsize=12, fontweight="bold", loc="left")
    ax_a.legend(fontsize=8, ncol=4, loc="upper right",
                framealpha=0.9)
    ax_a.set_ylim(0, max(0.95, ax_a.get_ylim()[1]))
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # ─── Panel B: Heatmap of cosine distance matrix (temp=0.0, exemplar) ───
    # Build a 4×4 creature similarity matrix averaged across all scenarios
    ax_b = fig.add_subplot(gs[1, 0])
    res_t0 = output["results_by_temperature"]["0.0"]
    avg_matrix = np.zeros((n_creatures, n_creatures))
    count_matrix = np.zeros((n_creatures, n_creatures))
    for sname in scenario_names:
        pairs = res_t0["per_scenario"][sname]["pairs"]
        for p in pairs:
            pair_str = p["pair"]
            # Parse pair names
            ci = next(i for i, c in enumerate(creature_labels)
                      if pair_str.startswith(c))
            cj = next(j for j, c in enumerate(creature_labels)
                      if pair_str.endswith(c) and j != ci)
            d = p["bear"]["cosine_dist"]
            avg_matrix[ci][cj] += d
            avg_matrix[cj][ci] += d
            count_matrix[ci][cj] += 1
            count_matrix[cj][ci] += 1

    count_matrix[count_matrix == 0] = 1
    avg_matrix /= count_matrix

    im = ax_b.imshow(avg_matrix, cmap="YlOrRd", vmin=0, vmax=0.5)
    ax_b.set_xticks(range(n_creatures))
    ax_b.set_yticks(range(n_creatures))
    ax_b.set_xticklabels(creature_labels, fontsize=10)
    ax_b.set_yticklabels(creature_labels, fontsize=10)
    for i in range(n_creatures):
        for j in range(n_creatures):
            v = avg_matrix[i][j]
            color = "white" if v > 0.3 else "black"
            ax_b.text(j, i, f"{v:.3f}", ha="center", va="center",
                      fontsize=10, fontweight="bold", color=color)
    ax_b.set_title("(b)  Inter-Creature Cosine Distance\n"
                    "(BEAR, temp=0.0, avg across scenarios)",
                    fontsize=11, fontweight="bold", loc="left")
    plt.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04,
                 label="Cosine Distance")

    # ─── Panel C: Per-scenario breakdown (cosine dist, temp=0.0) ───
    ax_c = fig.add_subplot(gs[1, 1])
    n_pairs = n_creatures * (n_creatures - 1) // 2
    pair_labels_full = []
    for i in range(n_creatures):
        for j in range(i + 1, n_creatures):
            pair_labels_full.append(
                f"{creature_labels[i]}–{creature_labels[j]}")

    x_sc = np.arange(len(scenario_names))
    bar_w_sc = 0.7 / n_pairs
    cmap = plt.cm.Set2
    for pi in range(n_pairs):
        vals = []
        for sname in scenario_names:
            pairs = res_t0["per_scenario"][sname]["pairs"]
            vals.append(pairs[pi]["bear"]["cosine_dist"])
        offset = (pi - n_pairs / 2 + 0.5) * bar_w_sc
        ax_c.bar(x_sc + offset, vals, bar_w_sc,
                 label=pair_labels_full[pi],
                 color=cmap(pi / max(n_pairs - 1, 1)), alpha=0.85,
                 edgecolor="white", linewidth=0.3)

    ax_c.set_xticks(x_sc)
    ax_c.set_xticklabels(
        [scenario_display.get(s, s) for s in scenario_names],
        fontsize=8)
    ax_c.set_ylabel("Cosine Distance", fontsize=10)
    ax_c.set_title("(c)  Per-Scenario Cosine Distance by Pair\n"
                    "(BEAR, temp=0.0)",
                    fontsize=11, fontweight="bold", loc="left")
    ax_c.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.9)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # ─── Panel D: Per-scenario Hausdorff (temp=0.0) ───
    ax_d = fig.add_subplot(gs[2, 0])
    for pi in range(n_pairs):
        vals = []
        for sname in scenario_names:
            pairs = res_t0["per_scenario"][sname]["pairs"]
            vals.append(pairs[pi]["bear"]["hausdorff"])
        offset = (pi - n_pairs / 2 + 0.5) * bar_w_sc
        ax_d.bar(x_sc + offset, vals, bar_w_sc,
                 label=pair_labels_full[pi],
                 color=cmap(pi / max(n_pairs - 1, 1)), alpha=0.85,
                 edgecolor="white", linewidth=0.3)

    ax_d.set_xticks(x_sc)
    ax_d.set_xticklabels(
        [scenario_display.get(s, s) for s in scenario_names],
        fontsize=8)
    ax_d.set_ylabel("Hausdorff Distance", fontsize=10)
    ax_d.set_title("(d)  Per-Scenario Hausdorff Distance by Pair\n"
                    "(BEAR, temp=0.0)",
                    fontsize=11, fontweight="bold", loc="left")
    ax_d.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.9)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    # ─── Panel E: Delta (BEAR − Static) across scenarios ───
    ax_e = fig.add_subplot(gs[2, 1])
    for ti, (t_key, marker, ls) in enumerate(
            [("0.0", "o", "-"), ("0.7", "s", "--")]):
        res = output["results_by_temperature"][t_key]
        delta_cos = []
        delta_hd = []
        for sname in scenario_names:
            sc = res["per_scenario"][sname]
            delta_cos.append(
                sc["bear_mean"]["cosine_dist"] -
                sc["static_mean"]["cosine_dist"])
            delta_hd.append(
                sc["bear_mean"]["hausdorff"] -
                sc["static_mean"]["hausdorff"])
        ax_e.plot(x_sc, delta_cos, marker=marker, ls=ls, color=C_BEAR,
                  linewidth=2, markersize=7,
                  label=f"Δ Cosine (temp={t_key})")
        ax_e.plot(x_sc, delta_hd, marker=marker, ls=ls, color="#7B1FA2",
                  linewidth=2, markersize=7,
                  label=f"Δ Hausdorff (temp={t_key})")

    ax_e.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax_e.set_xticks(x_sc)
    ax_e.set_xticklabels(
        [scenario_display.get(s, s) for s in scenario_names],
        fontsize=8)
    ax_e.set_ylabel("Δ (BEAR − Static)", fontsize=10)
    ax_e.set_title("(e)  BEAR Advantage per Scenario\n"
                    "(positive = BEAR more diverse)",
                    fontsize=11, fontweight="bold", loc="left")
    ax_e.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.9)
    ax_e.spines["top"].set_visible(False)
    ax_e.spines["right"].set_visible(False)

    fig.suptitle(
        "Genome-Conditioned LLM Output Diversity\n"
        f"Model: {output['metadata']['llm_model']}  |  "
        f"4 creatures × 5 scenarios",
        fontsize=14, fontweight="bold", y=0.97)

    chart_path = OUT_DIR / "eval6b_llm_dialogue.png"
    plt.savefig(chart_path, dpi=200, facecolor="white")
    plt.close()
    print(f"Chart saved to {chart_path}")


if __name__ == "__main__":
    main()
