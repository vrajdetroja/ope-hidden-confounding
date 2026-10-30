# Breaking Point: When Hidden Confounding Flips Off-Policy Evaluation

Off-policy evaluation (OPE) under hidden confounding, studied on the sepsis simulator of Oberst &
Sontag (2019). A hidden variable of known strength influences the clinicians' antibiotic decisions
and the patients' transitions. The simulator is ported to an exact transition tensor, so true
policy values, the confounding strength Γ* and the infinite-data limit of every estimator are
computed exactly.

## Reproduce

```bash
make setup     # .venv with the pinned dependencies
make test      # unit tests, under a minute
make validate  # slow: port vs the original simulator, exact DP vs 10^6 Monte Carlo episodes
make smoke     # 2 cells x 3 replicates, under a minute
make sweep     # the full grid, resumable (about 35 hours on an 8-core laptop)
make sweep-r1  # regime R1 only
make figures   # every figure and table from results/
```

## Repository layout

| Path | Contents |
| --- | --- |
| `src/ope_conf/sim/` | simulator port, confounding regimes, policies |
| `src/ope_conf/truth.py` | exact values, Γ*, infinite-data limits |
| `src/ope_conf/data.py` | episode generation, with recorded and hidden columns kept apart |
| `src/ope_conf/estimators.py`, `msm.py`, `bounds.py` | nominal estimators and sensitivity bounds |
| `src/ope_conf/alarm.py`, `learn.py` | breakdown Γ and benchmark Γ̂; learned policies |
| `src/ope_conf/run.py` | one cell in, result rows out |
| `scripts/sweep.py` | parallel, resumable sweep over a grid in `configs/` |
| `scripts/figures.py`, `design_frontier.py`, `oracle_wis.py` | figures and tables |
| `tests/` | unit tests and validation against the original simulator |
| `third_party/sepsis/` | the original sim-v2 code, used only by the tests |

## License

MIT. The vendored simulator in `third_party/sepsis/` keeps its own MIT license.
