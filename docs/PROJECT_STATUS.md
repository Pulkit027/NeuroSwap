# NeuroSwap Project Status

## Current milestone: Rule-based Controller

Completed in this milestone:

- Configurable C++ baseline workload with sequential and strided read/write patterns.
- Deliberate initialization that touches all allocated pages.
- POSIX measurement foundation using `std::chrono` and `getrusage(RUSAGE_SELF)`.
- Baseline executable output for wall-clock time, page-fault deltas, and process maximum RSS.
- Minimal workload verification test wired into CTest.
- A raw output file from a real WSL2 baseline run in `results/raw/`.
- Pinner module with explicit `mlock()`/`munlock()` results and `RLIMIT_MEMLOCK` pre-checking.
- Separate `neuroswap_pinned` executable using the same configurable workload as the baseline.
- Pinner tests for invalid regions, zero sizes, an over-limit request, and a small lock/unlock attempt.
- Real WSL2 pinned run recorded in `results/raw/pinned_first_run.txt`: 16 MiB lock succeeded and was subsequently unlocked; observed `RLIMIT_MEMLOCK` soft limit was 64 MiB.
- File-backed Prefetcher with explicit `mmap()`, `munmap()`, and `madvise(MADV_WILLNEED)` results.
- Matched no-hint and hinted read workloads using one deterministic 16 MiB file, three iterations, and sequential access; raw results are stored in `results/raw/prefetch_first_run.txt`.
- Latest single-run observation: no-hint 445.026 ms, 512 minor faults, 1 major fault; `MADV_WILLNEED` 227.895 ms, 512 minor faults, 1 major fault. This is raw WSL2 evidence, not a general performance claim.
- Rule-based Controller connects the profiler, MemoryWorkload/Pinner, and file-backed Prefetcher paths. It selects baseline, pinned, or prefetched mode from user-configured workload type and access pattern; it does not predict pressure or access internal LLM/KV-cache memory.
- Controller demonstration command and actual output are stored in `results/raw/controller_first_run.txt`.
- Latest controller demo: `./build/neuroswap_controller --type sensitive --memory-mb 16 --iterations 3 --pattern sequential` selected `PINNED`; `mlock()` succeeded for 16 MiB and the workload recorded 160.818 ms, 4097 minor faults, 0 major faults, and 19964 KiB maximum RSS.
- Benchmark suite completed with five repetitions per configuration. Baseline and Pinner used 16 MiB, three iterations, sequential access. The paired file-backed experiment used a deterministic 16 MiB file, three iterations, and sequential access, recording both no-hint and `MADV_WILLNEED` cases per run.
- Raw benchmark outputs: `results/raw/baseline_run_01.txt` through `baseline_run_05.txt`, `pinned_run_01.txt` through `pinned_run_05.txt`, and `prefetch_run_01.txt` through `prefetch_run_05.txt`. Processed outputs: `results/processed/benchmark_summary.csv` and `results/processed/benchmark_statistics.csv`.
- Five-run means: baseline 72.281 ms; pinned 71.270 ms; file-backed no-hint 143.412 ms; file-backed `MADV_WILLNEED` 150.735 ms. The aggregate sample does not establish a guaranteed prefetch performance improvement.
- No graphs were generated because Matplotlib is not installed and no packages were added.

Not implemented:

- ML, LLM integration, `userfaultfd`, or `perf`

## Environment note

The workload uses Linux/POSIX measurement APIs and is intended to be built and run inside WSL2. Results are WSL2/Linux experimental observations, not native bare-metal Linux results. Timing can vary with host scheduling and virtualization. `mlock()` is bounded by the current process `RLIMIT_MEMLOCK` and does not override Linux memory management. The Prefetcher runs under `/mnt/c`; filesystem and page-cache behavior may differ from native Linux. Each case requests `posix_fadvise(POSIX_FADV_DONTNEED)` before mapping to reduce cache carryover, but that request is advisory and cannot guarantee a cold cache. `MADV_WILLNEED` is also advisory and does not guarantee page preloading or performance improvement.
