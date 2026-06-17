# nvfp4bench — Design Document

A command-line utility to measure the **peak achievable NVFP4 tensor-core performance** on the
NVIDIA DGX Spark (GB10 Grace Blackwell superchip), and to report it against NVIDIA's headline
claim of **1 PFLOP NVFP4**.

---

## 1. Target hardware

| Property | Value | Notes |
|---|---|---|
| Superchip | NVIDIA GB10 Grace Blackwell | 20-core Arm CPU + Blackwell GPU |
| GPU arch | Blackwell, **compute capability 12.1 (`sm_121`)** | Distinct ISA from datacenter Blackwell `sm_100` |
| SMs | **48** | 5th-gen Tensor Cores + RT Cores |
| CUDA cores | 6,144 | |
| Max GPU clock | **~2.42 GHz** | Used for theoretical-peak calc |
| Unified memory | 128 GB LPDDR5x | CPU+GPU coherent |
| Memory bandwidth | **273 GB/s** | The key constraint — see §3 |
| CUDA | 13.0 | Driver 580.159.03 |

### Peak NVFP4 figures
- **1 PFLOP** is the **sparse** (2:4 structured) FP4 number.
- **Dense** FP4 peak is **~500 TFLOPS** (half of sparse).
- Published SM121-tuned CUTLASS NVFP4 GEMM has reached **~356 TFLOPS dense** (≈71% of dense peak).
  That is our realistic baseline to match and beat.

**Theoretical dense peak sanity check:**
`500 TFLOPS / (48 SM × 2.42 GHz × 2 [FMA]) ≈ 2150 FP4 MACs / SM / cycle` — consistent with
5th-gen tensor-core throughput. The utility computes this peak at runtime from the *measured* clock
(via NVML) rather than hardcoding it.

### Headline measured result (the 1 PFLOP is real — but instruction-gated)

Measured on real GB10 silicon with a memory-free, register-resident MMA throughput
microbenchmark (`src/peak.cu`, the `--peak` mode; standalone `src/peak_mma.cu`). Each
rung issues independent back-to-back warp MMAs so the number is the raw tensor-core
issue rate, free of the L2/bandwidth ceiling that caps any real GEMM:

| Rung | Instruction | Format | K/instr | Measured | vs spec |
|---|---|---|---|---|---|
| 1 | `mma.sync … kind::mxf8f6f4` | byte-padded FP4, **1 code/byte** | 32 | **255.5 TFLOPS** | 51% of 500 |
| 2 | `mma.sp … kind::mxf8f6f4` | + 2:4 structured sparse | 64 | **511.1 TFLOPS** | 51% of 1000 |
| 3 | `mma.sync … kind::mxf4nvf4` | **native packed FP4, 2 codes/byte** | 64 | **511.1 TFLOPS** | **102% of 500** |
| 4 | `mma.sp … kind::mxf4nvf4` | **packed + 2:4 sparse** | 128 | **1022.2 TFLOPS** | **102% of 1000** |

The ladder is a clean `256 → 512 → 512 → 1022`: **packed = 2.00×, sparse = 2.00×,
combined = 4.00×**. Conclusions:

- **NVIDIA's 500 TFLOPS dense and 1 PFLOP sparse NVFP4 figures are real silicon on
  GB10** — both top rungs hit ~102% of spec. They are *not* marketing rounding.
- **But they are reachable only via the *native packed* `kind::mxf4nvf4` instruction
  (2 codes/byte, `m16n8k64` dense / `m16n8k128` sparse) — and, for the PFLOP, 2:4
  sparsity on top.** The byte-padded `kind::mxf8f6f4` path that most hand-written code
  (and our own first kernel) starts from tops out at exactly **half** rate (256/512),
  because it moves one FP4 code per byte and so issues half the K per instruction.
- This resolves the earlier puzzle that CUTLASS's GEMM (375 TFLOPS, memory-bound) sat
  *above* our 256 `mxf8f6f4` microbench ceiling: CUTLASS uses the packed `mxf4nvf4`
  instruction. 375 was its **memory** ceiling, not the compute ceiling — the silicon's
  real dense compute peak is 511 TFLOPS.
- GB10 unexpectedly **does** expose the warp-level *sparse* block-scaled FP4 MMA
  (`mma.sp::ordered_metadata … block_scale`); it is not confined to datacenter
  `sm_100`/`tcgen05`. The `ue8m0` scale type caps at `scale_vec::2X` (block-32);
  `scale_vec::4X` requires `ue4m3` (block-16) — the literal NVFP4 microscale.

The practical caveat stands: the 1 PFLOP is a **pure tensor-core** number. Any real GEMM
on GB10 is throttled far below it by the 273 GB/s LPDDR5 bandwidth (see §3) — packed FP4
operands are 4× smaller than the byte-padded form, which helps, but the arithmetic
intensity wall remains.

### Validated packed GEMM (turning the peak into a *useful* result)

The peak ladder above is throughput-only (data-independent). To prove the packed path is
actually *usable*, we reverse-engineered the packed numeric model and fragment layout and
built a correctness-checked GEMM on them:

- **`mxf4_calib.cu`** pins the packed model: FP4 decode is true E2M1 (`0,.5,1,1.5,2,3,4,6`);
  the `ue8m0` block scale is `2^(E−127)` (so `0x7F` = unit, `0x81` = ×4); `scale_vec::2X`
  uses two scale bytes, each governing one 32-wide K-block of the k64; both nibbles of a
  byte are live, one per K-slot.
- **`mxf4_layout.cu`** extracts the packed fragment layout by one-hot probing every nibble
  and self-validates a random tile (max abs err 0). The map is closed-form
  (`g=lane>>2, t=lane&3`): A `row=2g+((p>>3)&1), k=16t+8·((p>>3)>>1)+(p&7)`;
  B `col=g, k=16t+p`; D `m=2g+(dreg>>1), n=2t+(dreg&1)`.
- **`gemm_warp_mxf4.cu`** / the `custom` backend: with the A-tile stored packed (8 codes per
  32-bit word) and the B-tile packed-transposed, the per-warp gather and the C store are
  byte-for-byte the dense byte-padded kernel's — only `BK` (32→64) and the MMA opcode
  change. Result: **PASS vs the FP32 oracle (max abs err 0) and ~112 TFLOPS at 2048³**
  (L2-resident) — **2.3×** the byte-padded warp kernel's ~48, from the same source
  structure. Still bandwidth-bound at large shapes, as expected.

---

## 2. What NVFP4 actually is

NVFP4 is NVIDIA's 4-bit block-scaled float format:
- **Element:** `E2M1` (1 sign, 2 exponent, 1 mantissa bit), packed two-per-byte.
- **Block scale:** one **FP8 `e4m3`** scale factor (SF) per **block of 16 elements**.
- **Global scale:** a single **FP32** tensor-wide scale.

This finer-grained scaling (block size 16, e4m3 scales) is what makes NVFP4 more accurate than MXFP4
(block size 32, e8m0 scales).

The tensor-core MMA instructions that consume this format are
`tcgen05.mma ... kind::mxf4nvf4` and are **only emitted when compiling for `sm_121a`** (the
arch-specific `a` variant). Plain `sm_121` will *not* unlock them — this is a hard requirement.

---

## 3. Why a compute-bound GEMM (and the bandwidth trap)

Peak tensor throughput is only observable from a **compute-bound** GEMM `C[M×N] += A[M×K]·B[K×N]`.

The bandwidth budget is brutal: to be compute-bound at 500 TFLOPS we need arithmetic intensity
`>> 500e12 / 273e9 ≈ 1831 FLOP/byte`. A naive single GEMM that streams operands from LPDDR5 will
measure **273 GB/s**, not tensor FLOPS.

Mitigations the benchmark relies on:
1. **Large, squarish matrices** (e.g. M=N=K = 4096–16384) so intensity is in the thousands.
2. **Looped / persistent execution** — run the same GEMM many times keeping operands resident so
   steady-state HBM traffic per FLOP → ~0 (operands stay hot in L2).
3. **L2 residency hints** (`cudaStreamSetAttribute` access policy window) to pin operands.

The benchmark also runs a **separate memory-bandwidth probe** so the report can prove the GEMM is
compute-bound (roofline context).

---

## 4. Two implementation tracks

### Track A — CUTLASS reference (correctness + baseline)
- **CUTLASS ≥ 4.2.1** (vendored as a submodule). Older versions restricted block-scaled FP4 MMA to
  `sm_100a` and broke on sm_121 (CUTLASS issues #2800, #2947).
- Block-scaled NVFP4 GEMM via the CollectiveBuilder for `sm_121a`.
- Purpose: (a) ground-truth correctness oracle, (b) a strong published-class baseline (~356 TFLOPS)
  the custom kernel must beat.

### Track B — Custom kernel (the hero deliverable)

**ISA correction (confirmed on hardware):** GB10 / sm_121 is *consumer* Blackwell. It does **NOT**
have `tcgen05`, Tensor Memory (TMEM), TMA multicast, or 2-SM (`cta_group::2`) MMA — those are
datacenter Blackwell (sm_100) only. NVFP4 on GB10 runs through the **GeForce-style warp-level
block-scaled MMA**: `mma.sync.aligned.m16n8k32...kind::mxf4nvf4` with **register accumulation**
(the CuTe `SM120_16x8x32_TN_VS` atom). The custom kernel must therefore be built around that, not
tcgen05/TMEM.

Hand-written NVFP4 GEMM kernel implementing every optimization that actually applies to GB10:

- **`-arch=sm_121a`** target (mandatory; the `a` suffix unlocks the block-scaled MMA — see §6).
- **Warp-level block-scaled MMA** `mma.sync...kind::mxf4nvf4`, accumulating in registers; scale
  factors fed via the per-block selector operands.
- **TMA** (`cp.async.bulk.tensor`) or `cp.async` to stage A, B, and scale-factor tiles into shared
  memory (no multicast on GB10).
- **Multi-stage `cp.async`/mbarrier pipeline** within the **101 KB** smem budget (the binding
  constraint on GB10 — a quarter of sm_100's 232 KB; tile size is smem-limited, not compute-limited).
- **Swizzled shared memory** to eliminate bank conflicts; **register-blocked** accumulator tiles.
- **Persistent kernel** with a tile scheduler, grid sized to 48 SMs; cluster fixed at **1×1×1**.
- **L2 residency + operand reuse** to stay compute-bound under 273 GB/s.
- **2:4 structured sparsity** variant (sparse block-scaled `mma.sp`) for the 1 PFLOP comparison.

**Tuning axes:** tile shape (M/N/K) within the 101 KB smem cap, `cp.async` pipeline depth, warp
count, scheduler policy. Realistic goal: approach the CUTLASS baseline (~386 TFLOPS peak measured);
beating a mature CUTLASS collective by hand is unlikely, so a tile-config **autotuner** over the
CUTLASS collective is the higher-ROI route to "highest performance".

### From-scratch FP4 warp-MMA: hardware bring-up findings (src/mma_probe.cu)

Empirically verified on GB10/sm_121a (single 16×8×32 tile, `mma_probe.cu`):

- **Build flag:** must use explicit `-gencode=arch=compute_121a,code=sm_121a`. The `-arch=sm_121a`
  shorthand silently drops the `a`, producing `.target sm_121`, where `block_scale` / `mxf8f6f4`
  are rejected by ptxas.
- **Instruction (works):**
  `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0`
  — real E2M1×E2M1 FP4 tensor-core matmul. (The dedicated NVFP4 `mxf4nvf4`/k64/ue4m3 instruction's
  exact operand encoding is undocumented in accessible sources; this MX-scaled FP4 path delivers the
  same FP4 tensor-core FLOPs.)
- **Operand registers (per lane):** A = 4×u32, B = 2×u32, C/D = 4×f32, plus one UE8M0 scale byte
  per side with `{byte_id, thread_id}` selectors (both 0 for scale_vec::1X).
- **FP4 packing:** byte-padded — one E2M1 code per byte (e.g. all-ones operand = `0x02020202`).
- **Numeric model (calibrated):** output element = `(Σ_{k=0..31} e2m1(a_k)·e2m1(b_k)) · 2^(Ea−129) · 2^(Eb−129)`,
  where `Ea,Eb` are the UE8M0 scale bytes. **Unit scale byte = 0x81** (not 0x7F); E2M1 decodes
  standard (0x2=1.0, 0x4=2.0, 0x1=0.5). At unit scale, all-ones gives exactly 32.0.

**SOLVED fragment layout** (reverse-engineered via one-hot probing + self-validation, `mma_probe.cu`,
exact match on random tiles). For lane `L` (warp), `g = L>>2`, `t = L&3`:

```
A operand (areg r in 0..3, byte p in 0..3):
    row = 2*g + (r & 1)            k = t*8 + (r>>1)*4 + p
B operand (breg r in 0..1, byte p in 0..3):
    col = g                        k = t*8 + r*4 + p
D accumulator (dreg d in 0..3):
    m   = 2*g + (d>>1)             n = t*2 + (d & 1)
```

So a warp computes one 16×8 output tile over k=0..31; rows are interleaved `{2g, 2g+1}`, columns
`{2t, 2t+1}`, K tiled as `t*8 + kgroup*4 + byte`. FP4 is byte-padded (one code per byte).

**Hardware FP4 decode (measured):** `value = code * 0.5` for codes 0..7 (0,0.5,1.0,...,3.5) — a
fixed-point reading, NOT textbook E2M1. UE8M0 unit scale byte = `0x81`.

These formulas drive the from-scratch warp-MMA GEMM (`gemm_custom.cu`, standalone
`gemm_warp.cu`), wired in as the **`custom`** backend of the benchmark.

**Optimization results (2048³, GB10):**

| version | TFLOPS | note |
|---|---|---|
| v0 naive global-gather | 6.1 | one MMA/warp, byte gather |
| register-blocked (16 MMAs/warp) | ~20→48 | hide MMA latency |
| + vectorized 32-bit shared reads | 46–48 | A row-major, B transposed |
| + software-pipelined double buffer | 48.0 | 1 sync/stage, load/compute overlap |

Final: **~48 TFLOPS at 2048³, 8× over naive, fully validated** (exact vs CPU
reference). Diagnostics showed the kernel is bound by the manual shared→register
fragment gather feeding the MMAs (double-buffering changed nothing → not
sync/load/occupancy bound). Breaking past this needs `ldmatrix`-based fragment
loads — see `TODO_ldmatrix.md`. The kernel uses byte-padded FP4 codes (`code*0.5`
decode, unit scale `0x81`) and generates its own operands; correctness is validated
standalone.

**Shape/L2 caveat:** the 48 TFLOPS is at 2048³, where both operands fit in the
24 MB L2, so the kernel's inefficient global loads are hidden. At the larger peak
shape `4096×14336×4096` the byte-padded B operand alone is **56 MB** (the
`mxf8f6f4` format stores one FP4 code per *byte* — 2× dense), spilling to DRAM and
making the kernel **memory-bound at ~24 TFLOPS**. The honest large-shape number is
~24 TFLOPS; the 48 was L2-flattered. Two fixes, both in `TODO_ldmatrix.md`: a
dense-packed (`mxf4nvf4`, 2 codes/byte) operand format halves the traffic, and
`ldmatrix` removes the gather bottleneck.

**Measured GB10 reality (this project):** dense NVFP4 peaks at **~386 TFLOPS (77% of the 500 spec)**
on short kernels (≤~1.3 ms, e.g. 4096×14336×4096); long square GEMMs (≥8192³) **thermally throttle**
to <100 TFLOPS. The benchmark reports both burst-peak (min time) and sustained (median).

---

## 5. CLI design

Binary: `nvfp4bench`

```
nvfp4bench [--kernel custom|cutlass|both]   (default: both)
           [--mode dense|sparse|both]        (default: both)
           [--m N --n N --k N | --sweep]     (size or preset sweep)
           [--iters N] [--warmup N]
           [--lock-clocks]                   (use nvidia-smi locked graphics clock)
           [--validate]                      (compare custom vs CUTLASS/FP32 ref)
           [--csv FILE] [--json FILE]
           [--bandwidth]                     (run the mem-bandwidth roofline probe)
```

**Run flow:**
1. **Probe** device — SM count, max/current clock (**NVML**, since `cudaDeviceProp.clockRate` is
   removed in CUDA 13), memory, compute capability; compute theoretical dense & sparse peak.
2. **Build operands** — random NVFP4 tensors + block/global scales.
3. **Warm up**, then time many iterations with **CUDA events**; report **median**.
4. **Validate** (optional) — dequantize to FP32 and compare against a reference GEMM; report max
   relative error (FP4 is lossy, so tolerance-based pass/fail).
5. **Report** — achieved TFLOPS, % of dense peak (~500), % of sparse peak (~1000), and efficiency
   vs the clock×SM theoretical number.

**Sample report:**
```
Device: NVIDIA GB10 (sm_121)  48 SM @ 2.42 GHz
Theoretical dense FP4 peak: 499.5 TFLOPS   | sparse: 999.0 TFLOPS
GEMM 8192x8192x8192  NVFP4 dense   custom : 392.1 TFLOPS  (78.5% dense)
GEMM 8192x8192x8192  NVFP4 dense   cutlass: 356.4 TFLOPS  (71.4% dense)
GEMM 8192x8192x8192  NVFP4 2:4     custom : 742.0 TFLOPS  (74.3% sparse)
Memory bandwidth (STREAM): 271 GB/s  -> GEMM is compute-bound ✓
Validation: max rel err 1.6e-2 (PASS, tol 5e-2)
```

---

## 6. CUDA 13 specifics & gotchas

- **`cudaDeviceProp::clockRate` removed** — get clock via **NVML**
  (`nvmlDeviceGetMaxClockInfo(NVML_CLOCK_SM)`); link `-lnvidia-ml`. Avoid the removed struct member.
- **Arch flag must be `sm_121a`** (not `sm_121`) or the NVFP4 MMA instructions are not generated.
- **CUTLASS ≥ 4.2.1** required for sm_121 block-scaled FP4.
- Verify `cta_group::2` (2-SM MMA) availability on GB10; fall back to `cta_group::1` if unsupported.
- Lock clocks during measurement for reproducibility; always report the clock used.

---

## 7. Build

- **CMake** + CUDA 13, `CMAKE_CUDA_ARCHITECTURES = 121a`.
- CUTLASS vendored as a git submodule (header-only include).
- Link `cuda`, `nvidia-ml` (NVML).
- Layout:
  ```
  nvfp4bench/
    CMakeLists.txt
    third_party/cutlass/          (submodule)
    src/
      main.cu                     CLI + orchestration
      device_probe.{cu,h}         NVML/CUDA device + theoretical peak
      nvfp4.{cu,h}                NVFP4 pack/quantize/dequantize + SF layout
      gemm_custom.cu              hero tcgen05/TMA/TMEM kernel
      gemm_cutlass.cu             CUTLASS reference path
      bench.{cu,h}                timing, sweep, reporting (CSV/JSON)
      bandwidth.cu                roofline mem-bw probe
    README.md
  ```

---

## 8. Risks / open questions

1. **sm_121 tcgen05 FP4 toolchain maturity** — historically buggy; pin a known-good CUDA 13 + CUTLASS
   combo and keep the CUTLASS path as a fallback if the custom kernel hits codegen issues.
2. **Exact NVFP4 scale-factor layout** — must match the swizzled SF layout the MMA expects; mirror
   CUTLASS's layout rather than reinventing.
3. **`cta_group::2` on GB10** — unknown; treat as a tuning axis with a `cta_group::1` fallback.
4. **Realistic ceiling** — expect ~70–85% of dense peak (≈356–425 TFLOPS) in practice; the full
   500 TFLOPS dense / 1 PFLOP sparse are hardware maxima rarely hit by real kernels.

---

## 9. Build order (proposed)

1. Scaffold CMake + device probe (NVML clock, peak calc) — verifiable on its own.
2. NVFP4 quantize/dequantize + FP32 reference GEMM (correctness oracle).
3. CUTLASS NVFP4 path → first real peak number + validation.
4. Custom kernel v0 (TMA + TMEM + single-stage MMA) → correctness.
5. Optimize custom kernel (pipeline, warp specialization, swizzle, persistent scheduler).
6. Add 2:4 sparse path.
7. Reporting (CSV/JSON), bandwidth roofline, clock locking, final tuning sweep.

---

### Sources
- [NVIDIA DGX Spark product page](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
- [The Register — GB10 superchip details](https://www.theregister.com/2025/08/27/nvidia_blackwell_gb10/)
- [Kubesimplify — DGX Spark: GB10, sm_121, NVFP4](https://blog.kubesimplify.com/day-3-the-dgx-spark-unpacked-gb10-unified-memory-sm-121-and-the-one-reason-this-hardware-exists)
- [CUTLASS Blackwell functionality docs](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md)
- [CUTLASS issue #2947 — enable FP4/tcgen05 for sm_121](https://github.com/NVIDIA/cutlass/issues/2947)
- [NVIDIA Forums — SM121 CUTLASS NVFP4 356 TFLOPS result](https://forums.developer.nvidia.com/t/sm121-cutlass-kernel-optimization-results-nvfp4-356-tflops-moe-grouped-gemm-on-dgx-spark/359960)
