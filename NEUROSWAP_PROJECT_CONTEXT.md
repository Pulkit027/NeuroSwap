# NeuroSwap: Proactive User-Space Memory Orchestration for LLM Inference on Constrained Edge Devices

**Course:** BCSE303P — Operating Systems Lab  
**Project:** NeuroSwap  
**Current Stage:** Review 2 — Implementation, Results, Screenshots, and Live Demonstration

---

## 1. Project Overview

NeuroSwap is an Operating Systems project that explores **application-aware memory orchestration for local LLM inference under constrained memory conditions**.

The central problem is that an LLM workload has memory-access patterns that are different from a conventional application. Model data can be accessed in a predictable layer-oriented sequence, while the KV-cache represents frequently reused conversational/context information. Under memory pressure, generic virtual-memory mechanisms do not have explicit knowledge of these application-level semantics.

NeuroSwap therefore investigates whether a **user-space memory orchestration layer** can improve memory behavior by:

- observing memory and execution behavior,
- keeping selected important memory resident when possible,
- proactively requesting pages before they are needed,
- releasing pages that are no longer useful,
- and using measured system behavior to make memory-management decisions.

The project is intentionally designed as a **user-space OS prototype** rather than a modification of the Linux kernel.

> **Important implementation principle:** Claims in the project documentation must match what is actually implemented and experimentally demonstrated. Proposed functionality must not be presented as completed functionality before it is tested.

---

# 2. Problem Statement

Local LLM inference can require substantial memory for:

- model weights,
- runtime buffers,
- KV-cache/context data,
- temporary computation,
- operating-system processes and other workloads.

When available memory becomes constrained, virtual-memory mechanisms can introduce additional page movement and I/O. This can increase latency and reduce throughput.

The project investigates the following question:

> **Can application-aware user-space memory orchestration reduce memory-related performance penalties for predictable LLM-like workloads under constrained memory?**

The project focuses on three controllable mechanisms:

1. **Profiling** — observe memory and execution behavior.
2. **Memory Pinning** — use `mlock()` where appropriate to keep selected pages resident.
3. **Prefetching / Page Advice** — use `madvise()` to communicate expected access patterns to the operating system.

A controller will eventually combine these mechanisms.

---

# 3. Important Technical Clarifications

The following points are part of the project design and must be treated carefully during implementation.

## 3.1 Linux memory management

Linux uses sophisticated virtual-memory mechanisms; it should not be described simply as "an LRU algorithm that blindly swaps everything."

For the project presentation, the simpler explanation may be used:

> Generic OS memory management does not understand the semantic importance of an LLM's model layers or KV-cache in the same way an application-aware orchestrator can.

The implementation should demonstrate this motivation through controlled experiments rather than relying on unsupported claims.

## 3.2 KV-cache

The project proposal identifies KV-cache residency as an important motivation.

However:

> Do not claim that the implementation directly identifies, pins, or controls an actual LLM KV-cache unless the code demonstrably does so.

If direct KV-cache control is not feasible, the project should use a controlled memory workload that reproduces relevant access/residency characteristics and clearly label it as a prototype or controlled experiment.

## 3.3 `mlock()`

`mlock()` can request that virtual-memory pages remain resident in RAM.

It is **not equivalent to permanently overriding the kernel memory manager**.

The implementation must account for:

- `RLIMIT_MEMLOCK`,
- available memory,
- permissions,
- failure conditions,
- and the fact that excessive pinning can itself be harmful.

## 3.4 `madvise()`

`madvise()` provides advice to the kernel about expected memory-access behavior.

The project may investigate:

- `MADV_WILLNEED`
- `MADV_DONTNEED`

The implementation must not describe these calls as guaranteed synchronous page loading or guaranteed page eviction. Their actual behavior must be measured and documented.

## 3.5 `userfaultfd`

`userfaultfd` is an advanced Linux mechanism that can allow user-space handling of certain page-fault events.

It is **optional** for the Review 2 implementation.

Do not make the project dependent on `userfaultfd` unless:

1. it works in the actual environment,
2. its required operations are permitted,
3. the implementation provides a meaningful benefit,
4. and the feature can be demonstrated reliably.

## 3.6 Machine Learning component

The original proposal includes a lightweight Linear Regression component that could use measured CPU-compute and storage-I/O characteristics to adjust prefetch behavior.

This is a proposed enhancement, not automatically a completed feature.

The ML component should only be implemented if:

- sufficient real experimental data is available,
- the prediction target is clearly defined,
- the model provides measurable value,
- and it does not distract from the core OS implementation.

A working OS prototype is more important than adding ML purely for presentation.

---

# 4. Review 1 Proposal

The Review 1 proposal introduced three primary modules:

### Module 1 — Profiler

Purpose:

- collect execution and memory-related measurements,
- establish baseline behavior,
- provide telemetry for later decisions.

Originally proposed technologies:

- Python/Bash
- `perf`
- Linux process/system statistics

### Module 2 — Pinner

Purpose:

- investigate keeping selected important memory resident,
- use `mlock()` for controlled experiments.

Implementation language:

- C/C++

### Module 3 — Prefetcher

Purpose:

- investigate proactive memory access/page advice,
- use `madvise()` such as `MADV_WILLNEED` and `MADV_DONTNEED`.

Implementation language:

- C/C++

### Controller

The eventual controller should combine information from the modules and make memory-management decisions.

---

# 5. Proposed System Architecture

```text
                         NeuroSwap
                             |
                    +--------+--------+
                    |                 |
                Profiler          Controller
                    |                 |
                    |        +--------+--------+
                    |        |        |        |
                    |      Pinner  Prefetcher  Policy
                    |        |        |
                    +--------+--------+
                             |
                    Application / Workload
                             |
                  Linux Virtual Memory System
                             |
                       RAM / Storage
```

The important design principle is:

> NeuroSwap operates in user space and communicates memory-access expectations to the operating system through supported mechanisms.

It does not require a custom Linux kernel for the core prototype.

---

# 6. Actual Development Environment

The project is currently being developed on:

- **Host:** Windows
- **Linux environment:** WSL2
- **Distribution:** Ubuntu 24.04.4 LTS
- **Kernel:** `6.18.33.2-microsoft-standard-WSL2`
- **Architecture:** x86_64
- **CPU:** Intel Core i7-8665U
- **CPU:** 4 physical cores / 8 logical CPUs
- **RAM visible to WSL:** approximately 7.7 GiB
- **Swap visible to WSL:** 2 GiB
- **G++:** 13.3.0
- **Python:** 3.12.3
- **Git:** 2.43.0
- **strace:** 6.8
- **perf:** currently unavailable for the exact WSL kernel
- **`mlock()` documentation/API:** available
- **`madvise()` documentation/API:** available
- **`userfaultfd` kernel symbols:** present
- **`ulimit -l`:** 65536 KB
- **`vm.swappiness`:** 60

### WSL limitation

Results obtained inside WSL2 must be described as **WSL2/Linux experimental results**, not automatically as native bare-metal Linux results.

The project should document the environment alongside every benchmark.

---

# 7. Review 2 Requirements

Review 2 requires:

1. **System Design and Architecture**
2. **Module Identification**
3. **Module Completion**
4. **Results and Analysis**
5. **Screenshots of All Outputs**
6. **Live Demonstration of the Project**

The project must therefore move beyond the Review 1 proposal and demonstrate actual implementation.

---

# 8. Review 2 Implementation Strategy

The implementation should proceed incrementally.

## Stage 1 — Baseline Memory Workload

Create a controlled C/C++ workload that:

- allocates memory,
- accesses memory in a known pattern,
- optionally creates memory pressure,
- records execution and memory-related measurements.

This establishes a reproducible baseline.

---

## Stage 2 — Profiler

Implement the Profiler module.

Possible measurements:

- execution time,
- maximum resident set size,
- minor page faults,
- major page faults,
- process memory statistics,
- CPU usage where practical,
- I/O statistics where available.

Because `perf` is currently unavailable for the exact WSL kernel, the initial profiler should use reliable alternatives such as:

- `getrusage()`,
- `/proc`,
- application-level timing,
- and other available Linux interfaces.

`perf` should be treated as optional rather than a hard dependency.

---

## Stage 3 — Memory Pinner

Implement a controlled `mlock()` experiment.

Compare:

```text
Baseline
   |
   +--> memory workload
   |
   +--> measurements
```

against:

```text
Pinned
   |
   +--> mlock()
   |
   +--> memory workload
   |
   +--> measurements
```

The experiment should report:

- amount of memory requested,
- amount successfully pinned,
- failures if any,
- execution time,
- relevant memory/page-fault measurements.

The project must respect the current 64 MB `RLIMIT_MEMLOCK` unless the environment is deliberately changed and documented.

---

## Stage 4 — Prefetcher

Implement a controlled memory-mapped workload using:

- `mmap()`
- `madvise(MADV_WILLNEED)`
- optionally `madvise(MADV_DONTNEED)`

Compare:

```text
Baseline
   |
mmap()
   |
sequential access
```

against:

```text
Prefetch
   |
mmap()
   |
MADV_WILLNEED
   |
sequential access
```

Measure the actual difference rather than assuming prefetching will always improve performance.

---

# 9. Stage 5 — NeuroSwap Controller

After the individual modules work independently, integrate them.

The controller should:

1. obtain telemetry from the profiler,
2. determine the current workload state,
3. decide whether a memory-management action is appropriate,
4. invoke the relevant pinner/prefetcher operation,
5. record the action and result.

A simple initial controller is preferable.

Do not introduce complex ML or predictive logic until the basic controller is reliable.

---

# 10. Stage 6 — LLM Integration

Only after the OS-level prototype works should LLM integration be attempted.

A possible integration target is `llama.cpp` with a small open model.

The purpose is to investigate whether the memory-management techniques can be demonstrated in a real local inference workload.

Potential measurements include:

- inference time,
- time-to-first-token where measurable,
- token generation rate,
- memory consumption,
- page faults,
- I/O behavior.

### Important

If direct access to internal KV-cache memory is not practical, do not fake it.

Instead, clearly separate:

### Controlled OS experiment

Demonstrates:

- memory pressure,
- pinning,
- prefetching,
- page behavior.

### LLM experiment

Demonstrates:

- applicability to local LLM workloads,
- memory/performance behavior,
- practical limitations.

---

# 11. Optional ML Component

The proposed ML component is a lightweight Linear Regression model.

Potential inputs:

- recent CPU compute time,
- recent storage-read latency,
- memory pressure indicators.

Potential output:

- a prefetch timing/decision parameter.

However, the implementation must first define:

1. What exactly is being predicted?
2. What data is collected?
3. How is training performed?
4. How is prediction evaluated?
5. Does prediction improve the baseline controller?

If these questions cannot be answered with real data, the ML component should remain documented as future work rather than being artificially added.

---

# 12. Benchmarking Plan

The benchmark should compare at least:

### Experiment A — Baseline

No NeuroSwap intervention.

### Experiment B — Pinner

Memory pinning enabled.

### Experiment C — Prefetcher

Prefetching/page advice enabled.

### Experiment D — Combined

Pinner + Prefetcher + Controller.

If LLM integration is successfully implemented:

### Experiment E — LLM workload

Compare the relevant configurations on a small reproducible model/workload.

Each experiment should have:

- identical workload parameters,
- documented environment,
- multiple runs where practical,
- raw measurements,
- processed measurements,
- averages and variability where appropriate.

Do not cherry-pick favorable runs.

---

# 13. Results Storage

Use:

```text
results/
├── raw/
├── processed/
└── figures/
```

### `raw/`

Original command outputs and measurements.

### `processed/`

CSV/JSON files prepared for analysis.

### `figures/`

Graphs such as:

- execution time comparison,
- page-fault comparison,
- memory usage,
- I/O behavior,
- throughput/latency comparison where relevant.

Every figure should identify the experimental configuration.

---

# 14. Evidence and Screenshots

Maintain:

```text
screenshots/
```

Capture actual screenshots of:

- compilation,
- successful module execution,
- baseline output,
- pinner output,
- prefetcher output,
- controller output,
- benchmark results,
- graphs,
- LLM demo if successfully integrated.

Never generate fake terminal screenshots.

---

# 15. Live Demonstration Plan

The live demo should be designed so that it can be completed reliably in approximately 5–7 minutes.

Suggested flow:

1. Explain the problem.
2. Show the architecture.
3. Run the baseline workload.
4. Show profiler measurements.
5. Run the pinner experiment.
6. Run the prefetcher experiment.
7. Run the integrated controller.
8. Show the measured comparison.
9. Explain limitations and future work.

A backup route should exist using previously generated real results in case a live experiment fails.

---

# 16. Proposed Repository Structure

```text
neuroswap/
│
├── README.md
├── AGENTS.md
├── CMakeLists.txt
│
├── src/
│   ├── profiler/
│   ├── pinner/
│   ├── prefetcher/
│   └── controller/
│
├── experiments/
│
├── benchmarks/
│
├── tests/
│
├── scripts/
│
├── results/
│   ├── raw/
│   ├── processed/
│   └── figures/
│
├── screenshots/
│
├── docs/
│
└── reference/
    ├── review1/
    └── planning/
```

---

# 17. Development Rules

The implementation team/AI coding agent must follow these rules:

1. Do not fabricate experimental results.
2. Do not fabricate benchmark numbers.
3. Do not create fake screenshots.
4. Do not claim a feature works until it has been compiled and tested.
5. Keep the project executable and reproducible.
6. Keep raw benchmark data.
7. Separate baseline results from NeuroSwap results.
8. Document the exact environment used.
9. Clearly distinguish WSL2 results from native Linux results.
10. Do not claim direct KV-cache manipulation unless actually demonstrated.
11. Do not describe `mlock()` as a complete replacement for the Linux memory manager.
12. Do not describe `madvise()` as a guaranteed page-loading or page-eviction mechanism.
13. Do not make `userfaultfd` mandatory unless it is tested and useful.
14. Do not make `perf` mandatory because it is currently unavailable for the exact WSL kernel.
15. Prefer simple, reliable OS experiments over unnecessary complexity.
16. Build and test one module at a time.
17. Do not implement the entire project in one uncontrolled step.
18. Preserve working modules when adding new functionality.
19. Use Git commits after meaningful milestones.
20. If a technical assumption is questionable, verify it before using it in the final presentation.
21. Do not add ML merely for presentation value; it must have a measurable purpose.
22. Never represent proposed/future work as completed implementation.

---

# 18. Project Success Criteria for Review 2

The minimum successful implementation should demonstrate:

- a reproducible memory workload,
- a working Profiler,
- a working `mlock()`-based Pinner experiment,
- a working `madvise()`-based Prefetcher experiment,
- measurable baseline vs modified behavior,
- an integrated controller where feasible,
- actual screenshots of outputs,
- actual benchmark data,
- analysis of results,
- a working live demonstration.

LLM integration and the ML predictor are valuable extensions, but they should not compromise the reliability of the core OS implementation.

---

# 19. Final Project Positioning

NeuroSwap should be presented as:

> **A user-space, application-aware memory orchestration prototype that investigates whether predictable workload information can be used with Linux memory-management interfaces to improve memory behavior for constrained local LLM inference.**

The project is not claiming to replace the Linux kernel memory manager.

Instead, it investigates whether user-space mechanisms such as:

- profiling,
- `mlock()`,
- `madvise()`,
- and controlled orchestration

can provide application-aware memory-management behavior that generic memory management does not explicitly model.

---

# 20. Current Development Status

At the beginning of Review 2 implementation:

- [x] Review 1 proposal completed
- [x] Project architecture proposed
- [x] WSL2 environment verified
- [x] C++ compiler verified
- [x] Python verified
- [x] Git verified
- [x] `strace` verified
- [x] `mlock()` available
- [x] `madvise()` available
- [x] `userfaultfd` kernel support observed
- [ ] Baseline workload implemented
- [ ] Profiler implemented
- [ ] Pinner implemented
- [ ] Prefetcher implemented
- [ ] Controller implemented
- [ ] Benchmark suite implemented
- [ ] Results collected
- [ ] Figures generated
- [ ] LLM integration tested
- [ ] ML component evaluated
- [ ] Review 2 screenshots prepared
- [ ] Review 2 presentation prepared
- [ ] Live demonstration validated

---

## Important

This document describes the **project design and implementation plan**.

It must not be treated as evidence that an implementation already exists.

Only mark a module as complete after the corresponding code has been implemented, compiled, executed, and verified.
