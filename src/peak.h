// peak.h - pure tensor-core peak NVFP4 measurement (no memory traffic).
#pragma once

// Runs the four-rung FP4 MMA throughput ladder on `device` and prints results:
//   mxf8f6f4 dense (k32) -> sparse (k64) -> mxf4nvf4 packed (k64) -> packed+sparse (k128)
// The top rung is the direct GB10 measurement of NVIDIA's 1 PFLOP NVFP4 figure.
// Returns the packed+sparse peak in TFLOPS (0 on failure).
double run_peak_ladder(int device);
