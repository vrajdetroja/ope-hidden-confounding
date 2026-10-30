"""Experiment configuration, loaded from YAML and hashed into every result row."""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import yaml

# Fields that define a cell of the design; the data seed depends on these only, so
# changing an analysis setting (say, the number of bootstrap draws) keeps the datasets.
CELL_FIELDS = ("regime", "lam", "w", "kappa", "kappa_p", "n", "H")


@dataclass(frozen=True)
class Config:
    # simulator
    H: int = 20
    gamma: float = 1.0
    p_diabetes: float = 0.2
    # hidden confounder
    regime: str = "R1"  # R1 single decision | R2 memoryless | R3 persistent (hidden diabetes)
    lam: float = 0.0  # tilt of the antibiotic decision (R1, R2)
    w: float = 0.0  # weight on the diabetes-aware clinician (R3)
    p_u: float = 0.3
    kappa: float = 0.5  # outcome channel: u = 1 and no antibiotics worsens HR and BP
    kappa_p: float = 0.0  # prognostic channel: u = 1 drops O2
    # policies
    epsilon: float = 0.1
    p_clip_lo: float = 0.05
    p_clip_hi: float = 0.95
    vi_discount: float = 0.99
    soften: float = 0.01
    rl_pessimism_c: float = 0.5
    # estimation
    n: int = 2000
    n_boot: int = 200
    smoothing: float = 0.0625  # total pseudo-count 0.5 per state, split over the 8 actions
    min_cell: int = 20
    gamma_hi: float = 20.0
    gamma_tol: float = 1e-3
    benchmark_min_count: int = 20
    benchmark_quantile: float = 0.95
    # bookkeeping
    root_seed: int = 20261102

    def __post_init__(self):
        if self.regime not in ("R1", "R2", "R3"):
            raise ValueError(f"unknown regime {self.regime!r}")
        if self.regime == "R3" and self.lam != 0:
            raise ValueError("R3 uses the mixing weight w, not lam")
        if self.regime != "R3" and self.w != 0:
            raise ValueError("w applies to R3 only")
        if not (1 <= self.H and 0 < self.n and 0 <= self.p_u <= 1 and 0 <= self.w <= 1):
            raise ValueError("invalid configuration")

    @property
    def cell_id(self) -> str:
        return "_".join(f"{k}={getattr(self, k)}" for k in CELL_FIELDS)

    def hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def seed_sequence(self, replicate: int) -> np.random.SeedSequence:
        """Seed for one dataset: spawned from the root seed by (cell, replicate), never reused."""
        cell_key = int(hashlib.sha256(self.cell_id.encode()).hexdigest()[:8], 16)
        return np.random.SeedSequence(entropy=self.root_seed, spawn_key=(cell_key, replicate))


def load_config(path: str | Path, **overrides) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    raw.update(overrides)
    return Config(**raw)


def expand_sweep(path: str | Path) -> tuple[str, int, list[Config]]:
    """Expand a sweep file into its cells. Returns (name, replicates, cells)."""
    path = Path(path)
    spec = yaml.safe_load(path.read_text())
    base = yaml.safe_load((path.parent / spec["base"]).read_text()) if "base" in spec else {}
    base.update(spec.get("overrides", {}))
    cells, seen = [], set()
    for grid in spec["grids"]:
        keys = list(grid)
        for values in itertools.product(*(grid[k] for k in keys)):
            cfg = Config(**{**base, **dict(zip(keys, values))})
            if cfg.cell_id not in seen:
                seen.add(cfg.cell_id)
                cells.append(cfg)
    return spec["name"], int(spec["replicates"]), cells


def git_sha() -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha[:12] + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def with_overrides(cfg: Config, **kw) -> Config:
    return replace(cfg, **kw)
