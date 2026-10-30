"""Run a sweep: every (cell, replicate) once, in parallel; resumable.

Each task is a chunk of replicates of one cell and writes its own parquet file under
<out>/_parts/, so an interrupted sweep picks up where it stopped. When all tasks are done the
parts are merged into <out>/results.parquet (one row per cell x replicate x policy x method).

    python scripts/sweep.py configs/sweep_main.yaml --out results/main [--only-regime R1] [--jobs -1]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from ope_conf.config import Config, expand_sweep
from ope_conf.run import run_replicate


def run_task(cfg: Config, reps: list[int], path: Path) -> tuple[str, int, float]:
    t0 = time.time()
    rows = [row for r in reps for row in run_replicate(cfg, r)]
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.rename(path)                                   # atomic: a part exists only when complete
    return cfg.cell_id, len(reps), time.time() - t0


def merge(out: Path) -> pd.DataFrame:
    parts = sorted((out / "_parts").glob("*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df.to_parquet(out / "results.parquet", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--only-regime", default=None)
    ap.add_argument("--only-n", type=int, default=None)
    ap.add_argument("--replicates", type=int, default=None, help="override the number of replicates")
    ap.add_argument("--chunk", type=int, default=5)
    args = ap.parse_args()

    name, replicates, cells = expand_sweep(args.sweep)
    replicates = args.replicates or replicates
    cells = [c for c in cells if (args.only_regime is None or c.regime == args.only_regime)
             and (args.only_n is None or c.n == args.only_n)]
    out = Path(args.out)
    (out / "_parts").mkdir(parents=True, exist_ok=True)
    tasks = []
    for cfg in sorted(cells, key=lambda c: c.n):       # cheap cells first
        for start in range(0, replicates, args.chunk):
            reps = list(range(start, min(replicates, start + args.chunk)))
            path = out / "_parts" / f"{cfg.hash()}_{start:04d}.parquet"
            if not path.exists():
                tasks.append((cfg, reps, path))
    print(f"[{name}] {len(cells)} cells x {replicates} replicates; {len(tasks)} chunks to run", flush=True)
    t0 = time.time()
    done = 0
    for cell_id, n_reps, dt in Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
            delayed(run_task)(*t) for t in tasks):
        done += 1
        print(f"  [{done}/{len(tasks)}] {cell_id} x{n_reps} in {dt:.0f}s "
              f"(elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)
    df = merge(out)
    print(f"merged {len(df)} rows into {out / 'results.parquet'}", flush=True)


if __name__ == "__main__":
    main()
