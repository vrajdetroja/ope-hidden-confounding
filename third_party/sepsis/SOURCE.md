# Vendored simulator (untouched)

`sepsisSimDiabetes/` is copied byte-for-byte from the `sim-v2` branch of
https://github.com/clinicalml/gumbel-max-scm at commit
`f923f7747ef00103337fa1e88dfad92b98db2993` (2020-02-06), MIT license (see `LICENSE`).

It is used **only by tests**, to validate the exact transition tensor in
`src/ope_conf/sim/dynamics.py` against the original sampling code.

sim-v2 differs from `master` by one line (`MDP.transition_fluctuate`): a non-diabetic
patient's glucose can now drift up to level 4 (`min(4, ...)`; master capped it at `min(1, ...)`).
