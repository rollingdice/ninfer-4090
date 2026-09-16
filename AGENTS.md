# AGENTS.md

These rules apply to the whole repository.

## Governing Objective & Engineering Priorities

Deliver maximum single-GPU inference performance, stability, and correctness for Qwen models on the NVIDIA GeForce RTX 4090 (`sm_89`, AD102).

1. **Functional and Numerical Correctness**: Mathematical fidelity comes first. Every floating-point operator must match its independent naive FP32/FP64 mathematical oracle.
2. **Architectural Quality and Ownership**: Maintain clear separation across core, artifact, ops, runtime, and serving layers. Never take sloppy shortcuts, duct-tape workarounds, or introduce code fragmentation.
3. **Hardware-Native Performance**: Optimize for Ada Lovelace (`sm_89`): 128 SMs, 72 MB L2 cache, 24 GB GDDR6X, Ada tensor cores (FP16/BF16 MMA), and vectorized 16-byte memory transactions.
4. **Reliability & Thread Safety**: Preserve strict boundaries on mutexes, non-blocking stream lifecycles, and CUDA Graph address stability.

---

## Product Contract & Target Platform

- **Target Hardware**: NVIDIA GeForce RTX 4090 (24 GB GDDR6X, AD102, 128 SMs), Windows 11, CUDA 13.x.
- **Supported Identities**: `qwen3.8-27b/groupwise-int`, `qwen3.6-27b/groupwise-int`.
- **KV Cache Quantization**: Uncompressed `int8`, `rk4v4-e8` (4-bit Conway-Sloane $E_8$ lattice), `rk2v4-e8` (2-bit $E_8$ cylinder).
- **Runtime Features**: MTP speculative decoding (K=1..15), D3D12/WDDM residency eviction management (`--wddm-evictable-budget`), DirectStorage 1.3 CoW state caching, CUDA Graph execution, and unified OpenAI/Anthropic HTTP serving (`ninfer-serve`).
- **Explicitly Excluded / Deleted**: Blackwell `sm_120`, NVFP4 formats, multi-GPU distributed orchestration, and dynamic plugin architectures.

---

## Repository Boundaries & Component Ownership

- `include/ninfer/`: In-tree public C++ Engine API (`engine.h`, `types.h`). Opaque interface used by CLI, server, and benchmarks.
- `src/core/`: Device memory allocation, `DeviceArena`, D3D12 residency management, DirectStorage 1.3 state caching, physical KV cache containers, and CUDA stream management.
- `src/artifact/`: Binary `.ninfer` container reader, parser, descriptor tables, and double-buffered DMA weight materializer.
- `src/ops/`: Mathematically closed operator implementations, fused kernels, and SIMT/MMA routines specialized for `sm_89`. Op ownership follows mathematical contracts, not target models.
- `src/targets/qwen3_6/`: Qwen3 family execution runtime, planning algorithms (`SequencePlan`, `RequestPlan`, `Program`), tokenizer, and chat templates.
- `src/serve/`: HTTP server, protocol translation (OpenAI `/v1/chat/completions`, Anthropic `/v1/messages`), and generation services.
- `apps/`: In-tree CLI (`ninfer.exe`) and server (`ninfer-serve.exe`).
- `bench/`: Standalone benchmarking binaries (`ninfer_bench.exe`).
- `tests/`: Unit, regression, and numerical oracle tests (CTest suite).

---

## Development & Verification Rules

1. **Strict Build Command Policy**: Never run build commands (`ninja`, `cmake`, etc.) unless explicitly instructed by the user.
2. **Mandatory CTest Verification**: For every new feature or critical change, implement corresponding CTests. All 84 CTests must pass with zero failures before completing work. On this Linux machine only 17 tests are CPU-only; when the GPU is busy run that CPU-only subset instead and note the restriction in the report.
3. **Numerical Verification**: Verify operator correctness against independent FP32/FP64 mathematical oracles. Pairwise parity against previous implementations is supplementary only.
4. **CUDA Architecture & Kernel Engineering Rules**:
   - **Stream Concurrency & Ordering**: Operations on legacy default Stream 0 do not synchronize with non-blocking streams (`cudaStreamNonBlocking`). Any allocation, touch-fill, or reset on Stream 0 must be explicitly drained (e.g. `cudaDeviceSynchronize` or stream events) before publishing pointers to non-blocking streams like `load_stream`.
   - **Wave Sizing & Grid Launch**: Size persistent and resident grids to live hardware SM counts (`device_sm_count()`, 128 SMs on RTX 4090) to guarantee full integer waves. Never hardcode SM counts from other architectures (e.g. 170 SMs), which creates unbalanced trailing straggler waves.
   - **L2 Cache Locality & CTA Rasterization**: When model weights exceed the 72 MB L2 cache, apply column-tile-major rasterization so resident CTAs sweep all column tiles of a row band concurrently, maximizing L2 cache hits and avoiding DRAM bandwidth degradation.
   - **Register Allocation vs Occupancy**: Per CUDA Best Practices, higher theoretical occupancy does not equate to higher performance. On memory-bound kernels, dynamically adjust `__launch_bounds__` to provide 128 registers per thread, eliminating spill stores/loads to local memory over chasing block occupancy.
   - **Vectorized Transactions**: Decoded weights and shared memory staging must use 16-byte vectorized transactions (`store_vec` / 128-bit stores) to minimize instruction count and shared memory bank pressure.
   - **Warp Shuffles vs Shared Memory Broadcast**: Avoid `__shfl_sync` with dynamic, non-uniform source lanes across sub-warp branches (which force compiler ABI branches and stack frames). Prefer direct `LDS` reads (hardware broadcast unit) when data is already resident in shared memory.
   - **Host Buffer Cacheability**: Host buffers read by CPU threads (e.g. `ordinary_host_egress`) must use cacheable pinned memory (`cudaMallocHost`), never write-combined memory (`cudaHostAllocWriteCombined`), avoiding severe uncached bus penalties.
   - **Instruction Cache Residency**: In multi-slab SIMT kernels, avoid uncontrolled full loop unrolling that bloats SASS beyond 32 KiB and exceeds the SM L1 instruction cache. Use `#pragma unroll 2` with software pipelining.
   - **Shared Activation Amortization**: In SIMT decode GEMMs where activations arrive as BF16 and require FP32 widening for `FFMA`, process two rows per CTA to share the widened activation and amortize global loads, cutting `PRMT` conversion instructions by 50% and global load sectors by ~48%.
   - **Whole-Program Device Optimization**: Keep separable compilation disabled (`-rdc=false`) to enable whole-program device optimization, cross-TU inlining, and global load pipelining across loop back edges.

> [!IMPORTANT]
> These are historically proven, empirically verified engineering invariants for this repository. However, never hesitate to call out sloppy coding, duct-taped workarounds, or architectural anti-patterns whenever reviewing or proposing code changes. Quality and hardware truth always take precedence over convenience.

---

## GPU Guardrail

- The RTX 4090 is **shared**: the user runs local models (LM Studio etc.) and desktop workloads on it.
- **Never start** `ninfer-serve`, `ninfer`, benchmarks, or GPU tests unless the user has explicitly said the GPU is free. Check `nvidia-smi` first.
- When the user asks for verification while the GPU is in use: run the CPU-only CTest subset and stop there. Report what could not be exercised.

## Working Context (this checkout)

- Branch: `feat/rtx-4090-sm89-native` (fork specialized for native `sm_89` single-GPU execution).
- Model artifact: `models/qwen3_8_27b.ninfer` (18.2 GB, sha256 `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e`).
- Serving: `./start.sh` → `ninfer-serve models/qwen3_8_27b.ninfer --kv-dtype rk4v4-e8 --spec mtp --draft-tokens 4 --lm-head-draft --max-context 240000 --preserve-thinking` (OpenAI `/v1/*` + Anthropic `/v1/messages` at `127.0.0.1:8080`).
- VRAM budget at 240k context: weights 16.67 GiB + KV 3.88 GiB, ~1.47 GiB free. Do not raise `--max-context` without re-checking headroom; the repo's "safe context" matrix is a Windows WDDM benchmark recommendation, not a guarantee on this Linux desktop.
- Measured reference points (single-request smoke, `rk2v4-e8`): prefill pp2048 ≈ 2,093 tok/s; decode MTP0 52.8, MTP2 95.2, MTP4 102.1, MTP7 98.4 tok/s. Workload benchmarks (`ninfer_bench`, MTP7): 216.9–229.9 tok/s.
- The worktree typically carries uncommitted in-flight changes (serve options, chat template, target registry). Never `git add -A` / commit / amend without an explicit request.

## Local Environment

- **This machine**: Ubuntu Linux (kernel 6.8), 24-core CPU, RTX 4090 (24 GB, driver 590.x).
- **Toolchain**: CUDA 13.3, GCC/G++ 13, CMake + Ninja.
- **CMake**: v1.2.0+ requires CMake ≥ 4.0 (`CMP0169` in `CMakeLists.txt`); system cmake 3.28.3 fails to configure. `build-linux-sm89/` is pinned to CMake 4.4.3 from the venv at `/tmp/cmake4venv` (`python3 -m venv /tmp/cmake4venv && /tmp/cmake4venv/bin/pip install "cmake>=4.0"`, then `cmake -S . -B build-linux-sm89` once if the venv is gone).
- **Build tree**: `build-linux-sm89/` (apps + tests enabled, benchmarks off):
  ```bash
  cmake --build build-linux-sm89 --target ninfer ninfer-serve
  ```
- **Upstream target**: Windows 11, MSVC x64, `build-ninja/` with `--split-compile=0` whole-program device compilation. Keep both paths building.

---

## Commits

Create commits only when explicitly requested by the user. Use Conventional Commit subjects:
- `feat`: User-facing features or new CLI/serving options.
- `perf`: Measured kernel, memory, or runtime optimizations.
- `fix`: Bug, race condition, or regression fixes.
- `test`: Adding or updating test suites.
- `build`: CMake, toolchain, or compiler configuration.
- `chore` / `docs`: Maintenance, cleanup, or documentation.
