"""Load a sim_log written by examples/evolutionary_ecosystem/server/app.py.

Handles both formats transparently:

- **Monolithic**: a single JSON file containing the full run's data
  (e.g. ``sim_log_diploid_codominant.json``).
- **Chunked**: when the simulation is launched with ``--chunk-size N``,
  the output rotates every N ticks into numbered files
  (``sim_log_diploid_codominant_01.json``,
  ``sim_log_diploid_codominant_02.json``, ...). Each chunk file is
  self-contained and holds the events from that tick window only.

``load_sim_log(path)`` accepts the *base* path (without the chunk suffix).
If a monolithic file exists at that path it is returned as-is. Otherwise
the loader globs ``<base>_NN.<ext>`` siblings, sorts them by chunk index,
concatenates the per-window lists (``birth_log``, ``death_log``,
``action_log``, ``snapshots``, ``epoch_snapshots``), and uses the *last*
chunk's ``metadata`` and ``final`` blocks (which reflect the cumulative
end-of-run state).

The returned dict has the same shape as a monolithic sim_log, so existing
analysis code that does ``data["birth_log"]`` etc. works unchanged.
"""

from __future__ import annotations

import json
import os
import re
from glob import glob
from pathlib import Path
from typing import Any


_CHUNK_RE = re.compile(r"_(\d+)\.json$")


def _chunked_paths(base_path: str) -> list[str]:
    """Return chunked sibling paths sorted by chunk index, or empty list."""
    base, ext = os.path.splitext(base_path)
    pattern = f"{base}_*{ext}"
    candidates = glob(pattern)
    indexed: list[tuple[int, str]] = []
    for p in candidates:
        m = _CHUNK_RE.search(p)
        if m:
            indexed.append((int(m.group(1)), p))
    indexed.sort()
    return [p for _, p in indexed]


def load_sim_log(path: str | Path) -> dict[str, Any]:
    """Load a sim_log from *path*, transparently merging chunks if present.

    Returns a dict with the same top-level keys as a monolithic sim_log:
    ``metadata``, ``final``, ``birth_log``, ``death_log``, ``action_log``,
    ``snapshots``, ``epoch_snapshots``.

    Raises FileNotFoundError if neither a monolithic file nor any chunks
    are found at *path*.
    """
    p = str(path)
    if Path(p).exists():
        with open(p) as f:
            return json.load(f)

    chunks = _chunked_paths(p)
    if not chunks:
        raise FileNotFoundError(
            f"No sim_log found at {p} (and no chunked siblings match {p}_NN.json)"
        )

    merged: dict[str, Any] = {
        "metadata": {},
        "final": {},
        "birth_log": [],
        "death_log": [],
        "action_log": [],
        "snapshots": [],
        "epoch_snapshots": [],
    }

    for chunk_path in chunks:
        with open(chunk_path) as f:
            data = json.load(f)
        for key in ("birth_log", "death_log", "action_log", "snapshots", "epoch_snapshots"):
            merged[key].extend(data.get(key, []))
        # Use the last chunk's metadata + final as the canonical end-of-run state.
        if data.get("metadata"):
            merged["metadata"] = data["metadata"]
        if data.get("final"):
            merged["final"] = data["final"]

    merged["metadata"].setdefault("chunked_from", chunks)
    return merged


__all__ = ["load_sim_log"]
