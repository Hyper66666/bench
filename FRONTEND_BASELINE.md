# Frontend Refactor Baseline (Frozen)

Reference sample:

- `bench/results/1771218708000-advanced-pipeline.json`

Frozen on:

- 2026-02-16

## Frontend-heavy checkpoints

| Profile | Frontend (ms) | Target (ms) | Delta vs target (ms) |
|---|---:|---:|---:|
| `scale_curve/100000` | 645.14 | 300.00 | +345.14 |
| `reachability_matrix/all_reachable` | 908.54 | 300.00 | +608.54 |
| `reachability_matrix/half_reachable` | 800.16 | 300.00 | +500.16 |
| `reachability_matrix/library_entryless` | 910.73 | 300.00 | +610.73 |

## Notes

- This file is the KPI baseline for `refactor-sgc-frontend-phase-architecture`.
- Perf gate now treats `reachability_matrix/all_reachable` frontend budget as a hard KPI in `hard` mode.
