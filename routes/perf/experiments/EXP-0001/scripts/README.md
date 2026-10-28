# perf/EXP-0001 helper scripts

Frozen plan: labbook `d6e3a243de880df22194457397d382906773ff0f`.
Measured code: `faa56e2b5c5c5dee60eb5dd7b6251944a21a6104`.

- `serve.sh` — TP8/EP8 native-default backends, radix off, bind 127.0.0.1.
- `stop_serve.sh` — kill this experiment's process tree only.
- `run_exp0001.py` — identity, serve, A/B/C/D timing then profiles.

Does not modify `/data/sgl_pr/worktrees/perf` production code.
