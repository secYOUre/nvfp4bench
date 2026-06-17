# TODO: push the from-scratch FP4 GEMM past ~48 TFLOPS with `ldmatrix`

## Where we are
The hand-written warp-MMA kernel (`src/gemm_custom.cu`, standalone `src/gemm_warp.cu`)
plateaus at **~48 TFLOPS** (8× over the naive baseline, fully validated). The
optimization sweep established the bottleneck empirically:

- Register blocking (16 MMAs/warp) and vectorized 32-bit shared reads were the wins
  (6 → 19 → 46 → 48 TFLOPS).
- Bigger `BK`, smaller blocks, coalesced-B, shared-stride padding all regressed.
- **Double-buffering (one sync/stage + load/compute overlap) changed nothing** →
  the kernel is *not* sync-, load-, or occupancy-bound.

The remaining limiter is the **manual shared→register fragment gather** that feeds
the MMAs (~24 individual `uint32` loads per warp per stage). CUTLASS avoids this
with `ldmatrix`, which loads a whole MMA fragment in the tensor-core-native layout
in a single instruction.

## The idea
Replace the manual fragment gather with `ldmatrix.sync.aligned` (the `.b16`/`.x4`
and possibly `.trans` variants). This should cut the per-stage shared-load
instruction count dramatically and let the MMAs issue closer to back-to-back.
Realistic upside: ~100–150 TFLOPS (still short of the CUTLASS 375 peak, which also
uses cluster/TMA-class machinery not available here).

## Why it's a real dig (not a quick change)
`ldmatrix`'s thread→element mapping for this **byte-padded FP4 / SM120 block-scaled**
MMA is undocumented — exactly the kind of thing we already reverse-engineered for the
MMA fragment layout (see DESIGN.md "SOLVED fragment layout"). Plan of attack:

1. Reuse the `mma_probe.cu` methodology: one-hot / structured probes to learn how
   `ldmatrix` distributes loaded bytes across lanes/registers, and reconcile it with
   the MMA's A/B fragment layout we already have.
2. Confirm whether the FP4 codes (byte-padded, 1 code/byte) need a `.b8`-style load
   or are loaded as `.b16`/`.b32` tiles, and whether `.trans` is needed for B.
3. Restructure the shared tiles into the `ldmatrix`-friendly swizzled layout, then
   swap the gather loop for `ldmatrix` + the existing `mma.sync` (the MMA and its
   numeric model are already solved and don't change).
4. Re-validate against the CPU reference (the harness in `gemm_warp.cu`), then re-time.

## Also worth trying (smaller levers)
- `cp.async` for the A load (contiguous) to overlap global→shared with compute
  (limited benefit here since operands are L2-resident, but cheap to add).
- A dedicated NVFP4 `mxf4nvf4` (k64, ue4m3, dense 2-codes/byte) instruction instead
  of byte-padded `mxf8f6f4` — halves operand data movement, but its exact PTX/operand
  encoding is also undocumented and would need probing.
