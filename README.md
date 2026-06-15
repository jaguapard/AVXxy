# AVXxy pre-alpha
A small x86 SIMD-library allowing for portability of SIMD code between feature sets.
# Disclaimer
The library is very experimental pet-project of a single developer and is provided with no warranty of any kind. While best effort is made to test its functionality, fix broken implementations and have optimal performance, some configurations may still have bugs or suboptimal implementations chosen by the dispatcher.

# Key assumptions
The library and this readme assumes that you are already familiar with basic x86 SIMD concepts and [intrinsic functions](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html).

Currently, MSVC compiler generates very suboptimal code, while Clang is almost perfect with this library. The library targets Clang first, thus, the performance of the code using the library can be severely degraded if using MSVC compiler. Other compilers are not tested. Be sure to check output assembly for spurious memory copies if in doubt.

# Getting started
Copy the `include` folder to your project, and include `include/avxxy.h`. This will include all the other files of the library and provide list of aliases (typedefs) for all supported vector and mask types. The library is header-only (but not single header), so no additional actions are needed. The library is fully enclosed in `avxxy` namespace by default. You can change the namespace by changing the value of `AVXXY_NAMESPACE` preprocessor macro.

Code example:
```
#include "lib/avxxy/avxxy.h"
using namespace AVXXY_NAMESPACE;

f32x16 a = f32x16::iota(); //0, 1, 2, ..., 15
f32x16 b = 6; //6 is broadcast to all lanes (elements) of the vector
f32x16 c = a + b; //6, 7, ..., 21

//function calls over vectors
bitmask16 mask = 0xFFFE;
f32x16 d = maskz_mov(mask, c); //6, 7, ..., 20, 0 (last element was masked out by zero-masking)
i32x16 e = d+0.5; //implicit cast to ints. Numerical value is truncated (rounded towards zero).
//or more explicitly, same result:
i32x16 f = vcvt<int32_t>(d); //6, 7, ..., 20, 0 (since it's truncated, not rounded to nearest)

i32x4 g(2,3,4,5); //initialization by a list of scalars
int ibuf[256] = {555};
i32x8 h = load<i32x8>(ibuf); //loads vector from ibuf pointer
i32x8 i = load<i32x8>(ibuf, 0x55); //loads vector from ibuf pointer using zero-masking. In this case, even elements are masked off, meaning the return value is 0, 555, 0, 555, ..., 555
i32x8 j = load<i32x8>(ibuf, 0x32, e); //merge-masking. Lane at index 5 (1<<5 == 32) is loaded from ibuf), all others are passed through from e

//buf = some memory region
//operations on extended vectors
f32x64 k = gather<float, 64>(buf, load<i64x64>(&indices)); //unconditionally gather 64 floats at 64 bit indices loaded from &indices
//can replace gather<float, 64> with gather<f64x64> for same result
f32x16 l = vcast<f32x16>(k); //take lowest quarter of k
f32x32 m = k.hi(); //upper half of k 

//note that you can also cast unrelated vector types.
i32x4 casted = vcast<i32x4>(k); //only lowest 4 elements are reinterpreted (not converted!) from the k.
f32x16 casted2 = vcast<f32x16>(casted); //lanes 4 and higher have undefined values, but do not read input vector out of bounds 

i8x64 small_ind = i8x64(3,5,1,5,...,123); //random small values
float scatterTarget[1024] = {0};
bitmask32 mask2 = m < f32x32(...); //comparison into mask
scatter<6>(m, scatterTarget, small_ind, mask2); //overrides Scale with 6
```

# Features
_This section describes feature set planned for release. Pre-release versions may not have some of the features, or may not implement them fully. Features italized are completely unimplemented as of now_
- Implementation of subset of operations: 
	- unmasked: add, sub, mul, div, logic or, and, xor, not, shift left and right, floor, ceil, abs, min, max, comparisons (==, !=, >, >=, <, \<=), cross-lane permute of 1 and 2 registers (permutex(2)var intrinsic, function name permx(2)), unpacklo, unpackhi, sqrtd, sqrtf, FP16 <-> FP32 conversions. 
	- masked: mask(z)_mov, blend, compress, load, store, gather, scatter
	- meta-operations: conversion of vector elements to other type, casting (reinterpreting) vectors
- C++-style operators for vector types (`+, -, *, /, <<, >>, &, |, ^, ~`) and in-place variations of them.
- Signed/unsigned integer vector types disambiguation.
- Vectors of these scalar types: double, float, (u)int64_t, (u)int32_t, (u)int16_t, (u)int8_t
- Extended vector types. Each vector can have power-of-two size from 2 to 64 inclusive. These do not require vector size to be native to architecture, operations on too large vectors are emulated with smaller ones.
- Fallbacks for all operations, down to fully scalar path.
- Emulation of operations that don't exist in native instruction sets. Examples include:
	- Multiplication of 64-bit integers pre-AVX512-DQ
	- Integer division
	- Small integer operations: 8-bit multiplication, shifts
	- Gather/scatter with 8 and 16 bit indices or values
	- Any size for index argument for permutex(2)var


- _Compile-time detection of available instruction sets and adjustment of dispatching using it. The machinery is exposed in `FeatureSet.h`, so you can tweak manually for certain architectures (enable or disable some CPU feature flags)._
- Compile-time dispatch pipeline with fixed preference order for different instruction sets
- Interoperability with intrinsics: the vector and mask vector types smaller than 65 bytes and bit masks smaller than 65 bits are implicitly castable to and from their intrinsic versions (__m512, __m256i, __mmask16, etc), allowing them to be used mostly seamlessly inside intrinsic functions.
- `std::ostream&` operators for outputting vectors and masks as human-readable strings to output streams (console, files, etc).

# Further reading
[Technical details](docs/Technicals.md)