# Environment (Phase 1)

- Host: Windows 10, WSL2 Ubuntu 24.04.4, kernel 6.18.33.2-microsoft-standard-WSL2
- CPU: Intel i7-8665U (4 cores, AVX2); WSL limits: 6GB RAM, 4 CPUs, 2GB swap (.wslconfig)
- WSL disk location: D:\WSL\Ubuntu-24.04
- llama.cpp commit: 43fe9c64281ef735046adc025e9e7559a1f659a5 (CPU build, Release)
- Model: qwen2.5-1.5b-instruct-q4_k_m.gguf
- Model sha256: 6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e
- Verified: cgroup v2 memory limit writable, PSI available, drop_caches works
- First smoke run (1 run, not a benchmark): decode 10.69 tok/s, prefill 31.6 tok/s