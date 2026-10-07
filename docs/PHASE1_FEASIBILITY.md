# Phase 1 feasibility results (single runs, same prompt, -n 64, -c 2048, 4 threads)

All checks passed: cgroup v2 memory limit, PSI, drop_caches, llama.cpp CPU build, model inference.
These are single pilot runs, not benchmarks. Real experiments will repeat each condition.

| memory.max | decode tok/s | wall s | major faults | peak RSS MB |
|---|---|---|---|---|
| none  | 13.78 | 9.0  | 12   | 1781 |
| 1400M | 9.99  | 12.0 | 250  | 1401 |
| 1300M | 12.86 | 10.5 | 129  | 1301 |
| 1250M | 11.28 | 10.8 | 272  | 1252 |
| 1200M | 5.52  | 16.9 | 745  | 1202 |
| 1100M | 0.99  | 70.9 | 4043 | 1081 |
| 800M  | 0.81  | 86.1 | 4412 | 798  |

Finding: model thrashes below ~1200M because weights (1.1GB) cannot stay cached.
Candidate budget levels: none, 1200M (moderate), 1100M (severe, expensive per run).
Whether pin/prefetch help is NOT yet known.