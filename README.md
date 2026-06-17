# nvfp4bench

A command-line utility to measure **peak achievable NVFP4 performance** on the
NVIDIA DGX Spark (GB10 Grace Blackwell, `sm_121a`, CUDA 13), and report it against
NVIDIA's headline **1 PFLOP NVFP4** (sparse) / **~500 TFLOPS** (dense) figures.

See `DESIGN.md` for the full architecture and rationale.

## Layout

```
CMakeLists.txt
src/
  main.cu            CLI + orchestration
  device_probe.*     NVML clock + SM count + theoretical peak (CUDA 13 safe)
  nvfp4.*            E2M1/e4m3 codecs, block-scale packing, dequant
  reference.*        FP32 GEMM oracle for validation
  bench.*            timing, operand setup, CSV/JSON reporting
  bandwidth.cu       STREAM-triad bandwidth roofline probe
  gemm.h             unified backend interface
  gemm_stub.cu       on-the-fly dequant GEMM (correctness/bring-up, slow)
  gemm_cutlass.cu    CUTLASS NVFP4 block-scaled GEMM (trusted baseline, ~375 TFLOPS)
  gemm_custom.cu     from-scratch PACKED mxf4nvf4 warp-MMA GEMM (~112 TFLOPS) -- `custom`
  gemm_warp.cu       standalone byte-padded mxf8f6f4 kernel (~48 TFLOPS, validates + times)
  gemm_warp_mxf4.cu  standalone packed mxf4nvf4 kernel (~112 TFLOPS, validates + times)
  peak.cu / peak.h   pure tensor-core MMA throughput ladder -- the `--peak` mode
  mma_probe.cu       hardware RE: byte-padded mxf8f6f4 MMA semantics + fragment layout
  mxf4_calib.cu      hardware RE: packed mxf4nvf4 decode + ue8m0 scale + nibble model
  mxf4_layout.cu     hardware RE: packed mxf4nvf4 fragment layout (self-validating)
  peak_mma.cu        standalone 4-rung peak ladder (mxf8f6f4/mxf4nvf4, dense+sparse)
  sparse_probe.cu    confirms GB10 exposes the warp-level sparse FP4 MMA
  mxf4_probe.cu      confirms the native packed mxf4nvf4 (k64) MMA
  mxf4_sparse_probe.cu  confirms packed+sparse mxf4nvf4 (k128) -- the 1 PFLOP instruction
TODO_ldmatrix.md     plan to push the custom kernel further via ldmatrix
```

## Key findings (GB10 / DGX Spark)

- **The 1 PFLOP is real silicon — and we measured it.** The memory-free MMA
  throughput microbench (`--peak`) reaches **1022 TFLOPS** packed+2:4-sparse NVFP4
  (102% of the 1 PFLOP spec) and **511 TFLOPS** packed dense (102% of the 500 spec).
  The full ladder is `256 → 512 → 512 → 1022` (packed 2×, sparse 2×, combined 4×).
- **The catch is the instruction, not the silicon.** The headline numbers need the
  *native packed* `kind::mxf4nvf4` MMA (2 codes/byte: `m16n8k64` dense, `m16n8k128`
  sparse). The byte-padded `kind::mxf8f6f4` path most hand-written code starts from
  runs at exactly **half** rate (256/512) — one code per byte, half the K per instr.
- **GB10 *does* expose the warp-level sparse block-scaled FP4 MMA** (`mma.sp …
  block_scale`) — not confined to datacenter `sm_100`/`tcgen05`. (`ue8m0` scales cap
  at `scale_vec::2X`; `4X` needs `ue4m3`.)
- **Real GEMMs are bandwidth-bound far below this.** CUTLASS peaks at **~375 TFLOPS**
  dense (`4096×14336×4096`) — its *memory* ceiling, not the compute ceiling. Long
  square GEMMs (≥8192³) also thermally throttle, so the GEMM path reports both **burst
  peak** (min time) and **sustained** (median).
- **GB10 has no `tcgen05`/TMEM/2-SM MMA** (those are datacenter sm_100). NVFP4 runs on
  the GeForce warp-level block-scaled `mma.sync...mxf4nvf4`/`mxf8f6f4`.
- The from-scratch `custom` kernel is built on a **fully reverse-engineered fragment
  layout + FP4 decode** for *both* the byte-padded `mxf8f6f4` and the native packed
  `mxf4nvf4` formats (undocumented; see DESIGN.md, `mma_probe.cu`, `mxf4_layout.cu`).
  It now uses the **packed** path and is validated at **~112 TFLOPS** (2048³,
  L2-resident) — 2.3× the byte-padded kernel, with bit-exact correctness vs the FP32
  oracle (`gemm_warp_mxf4.cu`, max abs err 0).

## Prerequisites

- CUDA 13.0 toolkit (driver 580.x), `nvcc` for `sm_121a`.
- A recent **CMake (>= 3.28 recommended)** — older CMake may not accept the
  `121a` accelerated architecture suffix.
- For the CUTLASS path: **CUTLASS >= 4.2.1** (earlier versions block FP4 on
  sm_121). Clone it into `third_party/cutlass`:
  ```
  git clone --depth 1 --branch v4.2.1 https://github.com/NVIDIA/cutlass third_party/cutlass
  ```

## Recommended bring-up order

The build is staged so each layer gives you something concrete to compile and
report back before the hard tensor-core code:

1. **Harness only** (no CUTLASS, custom v0 CUDA-core path):
   ```
   cmake -B build -DENABLE_CUTLASS=OFF -DENABLE_CUSTOM=ON
   cmake --build build -j
   ./build/nvfp4bench --kernel stub   --m 1024 --n 1024 --k 1024 --validate
   ./build/nvfp4bench --kernel custom --m 1024 --n 1024 --k 1024 --validate
   ./build/nvfp4bench --bandwidth
   ```
   Confirms device probe, NVFP4 packing, validation oracle, and bandwidth.

2. **Add CUTLASS** (trusted NVFP4 tensor-core baseline):
   ```
   cmake -B build -DENABLE_CUTLASS=ON
   cmake --build build -j
   ./build/nvfp4bench --kernel cutlass --m 4096 --n 14336 --k 4096
   ```
   Use the **4096×14336×4096** shape: it peaks on GB10 (~375 TFLOPS) and is the
   default. Large *square* shapes (e.g. 8192³) run long enough to thermally
   throttle to <100 TFLOPS — informative for the sustained vs burst distinction,
   but not the peak.

3. **Custom from-scratch kernel** (`--kernel custom`): a hand-written FP4 warp-MMA
   GEMM built on the reverse-engineered SM120 fragment layout. It uses the **native
   packed `mxf4nvf4`** path (2 codes/byte, k64). Build/validate standalone:
   ```
   nvcc -gencode=arch=compute_121a,code=sm_121a -O3 -o gemm_warp_mxf4 src/gemm_warp_mxf4.cu
   ./gemm_warp_mxf4      # validates 128^3 vs CPU ref (0 err), then times 2048^3 (~112 TFLOPS)
   ```
   The byte-padded predecessor (`gemm_warp.cu`, ~48 TFLOPS) and the hardware
   reverse-engineering that made both possible are alongside it: `mma_probe.cu`
   (byte-padded layout/decode), `mxf4_calib.cu` + `mxf4_layout.cu` (packed model and
   self-validating fragment-layout extraction).

## Usage

```
nvfp4bench [--kernel stub|cutlass|custom|all] [--mode dense|sparse|both]
           [--m N --n N --k N | --sweep] [--iters N] [--warmup N]
           [--validate] [--tol F] [--peak] [--bandwidth] [--device N]
           [--csv FILE] [--json FILE]
```

Measure the headline peak directly (pure tensor-core MMA ladder, no memory traffic —
this is where the 1 PFLOP shows up):
```
./build/nvfp4bench --peak
```

Example GEMM sweep with reporting:
```
./build/nvfp4bench --kernel all --mode both --sweep --csv results.csv
```

## Measurement notes

- Lock clocks for reproducible numbers: `sudo nvidia-smi -lgc <maxclock>` (the
  tool prints the max SM clock from NVML), and report the clock used.
- The benchmark targets a **compute-bound** regime; run `--bandwidth` to confirm
  the GEMM is not memory-limited at 273 GB/s.
- Expect roughly **70–85% of the 500 TFLOPS dense peak** in practice. The full
  1 PFLOP requires 2:4 sparsity *and* tolerates near-zero numerical accuracy.

## Toolchain gotchas (hard-won)

- **Use explicit `-gencode=arch=compute_121a,code=sm_121a`.** The `-arch=sm_121a`
  shorthand and some CMake versions silently drop the `a`, producing plain
  `sm_121` where the NVFP4 block-scaled MMA is rejected/aborts at runtime. The
  CMake build forces the explicit gencode.
- **CUTLASS ArchTag is `Sm120`** even on sm_121 (no `Sm121` tag exists); sm_121a is
  selected by the compile flag. Needs **CUTLASS ≥ 4.2.1**.
- `gemm_cutlass.cu` and `gemm_custom.cu` each define their CUTLASS/MMA kernel types
  in a single translation unit — defining the same `GemmUniversal` `__global__` in
  two TUs breaks `cudaFuncSetAttribute` registration at `initialize()`.

## Next step

See **`TODO_ldmatrix.md`** — pushing the custom kernel past ~48 TFLOPS means
reverse-engineering `ldmatrix`'s fragment mapping for this FP4 format.
