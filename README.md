# AVXxy pre-alpha
A small x86 SIMD-library allowing for portability of SIMD code between feature sets.

# Features
_This section describes feature set planned for release. Pre-release versions may not have some of the features, or may not implement them fully._
- Implementation of subset of operations: 
	- unmasked: add, sub, mul, div, logic or, and, xor, not, shift left and right, floor, ceil, abs, min, max, comparisons (==, !=, >, >=, <, \<=), cross-lane permute of 1 and 2 registers (permutex(2)var intrinsic, function name permx(2)), unpacklo, unpackhi, sqrtd, sqrtf, FP16 <-> FP32 conversions. 
	- masked: mask(z)_mov, blend, compress, load, store, gather, scatter
	- meta-operations: conversion of vector elements to other type, casting (reinterpreting) vectors
- Extended vector types. Each vector can have power-of-two size from 2 to 64 inclusive. Operations do not require the vector size to be native size to the architecture, operations on too large vectors are emulated with smaller ones.
- Fallbacks for all operations, down to fully scalar path.
- Emulation of operations that don't exist in native instruction sets:
	- Multiplication of 64-bit integers pre-AVX512-DQ
	- Integer division
	- Small integer operations: 8-bit multiplication, shifts
	- Gather/scatter with 8 and 16 bit indices or values

- Compile-time detection of available instruction sets and adjustment of dispatching using it. The machinery is exposed in `FeatureSet.h`, so you can tweak manually for certain architectures (enable or disable some CPU feature flags).
- Compile-time dispatch pipeline with fixed preference order for different instruction sets

# Further reading
[Technical details](docs/Technicals.md)