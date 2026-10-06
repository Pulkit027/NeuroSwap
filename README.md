# NeuroSwap

NeuroSwap is a user-space operating-systems prototype investigating application-aware memory orchestration for constrained LLM-like workloads.

## Current baseline

The project includes a reproducible baseline workload, a bounded `mlock()` Pinner experiment, a file-backed `mmap()`/`MADV_WILLNEED` Prefetcher experiment, and a rule-based Controller. It has no ML, LLM, or advanced kernel-facing functionality.

### Build in WSL2

```bash
g++ -std=c++17 -Wall -Wextra -Wpedantic -Iinclude \
  src/main.cpp src/workload/memory_workload.cpp src/profiler/measurement.cpp \
  -o build/neuroswap_baseline
```

### Run

```bash
./build/neuroswap_baseline --memory-mb 16 --iterations 3 --pattern sequential

./build/neuroswap_pinned --memory-mb 16 --iterations 3 --pattern sequential

./build/neuroswap_prefetch --file-mb 16 --iterations 3 --pattern sequential

./build/neuroswap_controller --type sensitive --memory-mb 16 --iterations 3 --pattern sequential
```

The Pinner reports `mlock()` failure instead of assuming memory was pinned; it is constrained by `RLIMIT_MEMLOCK` and does not override Linux memory management. The Prefetcher reports `madvise()` results but treats `MADV_WILLNEED` as advisory, not guaranteed preloading. Use `--help` to see supported options. `CMakeLists.txt` is included for systems with CMake installed.

The Controller is configuration-driven: `normal` selects baseline, `sensitive` selects bounded pinning, and `file-backed` with sequential access selects prefetching. It does not predict memory pressure or manipulate an LLM KV-cache.
