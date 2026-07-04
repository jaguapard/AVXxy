# Vector size notes
Vector sizes are limited to power of 2 sizes, starting from 1. Despite not having a hard cap (except size_t's maximum value), in practice, you will encounter more and more problems and limitations when using huge vectors:

- All masking operations are currently unavailable for vectors with more than 64 elements
- Some operations scale non-linearly with vector size, and/or can fall back to slower classes of instructions (permx(2) -> gather for instance)
- Some operations may not be supported at all for larger vectors (for example, conflict detection can't fit into 64 bit elements for N > 64)
- Kernels utilizing huge vectors will experience heavy register spilling
- Since all vectors don't perform any heap allocations, stack space can become a concern for huge vectors. 
- Intended user-range ends at 64 elements, with larger vectors only used as intermediate storage for example, for byte_shuffle indices

Previously (at and before commit 02872426 from July 3, 2026), vector sizes were hard capped at 64 elements. The limit was removed because it wasn't strictly necessary and caused many headaches with nothing being able to use larger vectors (some operations needed small vectors of same total byte size, i.e. intermediate results of vpopcnt). However, there are no plans for full-fledged support for huge vectors, and users are recommended to not exceed 64 elements soft-cap.

64 seems like a practical limit not requiring masking compications and cross-lane operation memory fallbacks. The masks fit into 64 bits, and vanilla x86 SIMD operations (as of now) don't need more than 64 lanes (64 lanes is the limit for zmm vectors containing 8-bit integers), while simple operations can be just broken down into chunks. Thus, u64x64 is the largest vector that can be returned by in-essence vanilla operations (converting 8-bit zmm to 64 bit integers or doubles)

# AVX512 fragmentation
According to [Wikipedia](https://en.wikipedia.org/wiki/AVX-512#CPUs_with_AVX-512), the minimal instruction set for AVX512 on somewhat desktop-related CPUs (earliest Skylake-X HEDT and Skylake-SP server CPUs) is: AVX512 F, CD, VL, DQ, BW. Only Xeon Phi had some of them missing, and currently, the support for these is out of the library's scope. Thus, the emulations may not be available and feature set boundaries may not be exactly respected with instructions inside this minimal set.

# AVX512 downclocking
Note that this library does not account for any possible CPU downclocking due to using "heavy" instructions (operating on 512 bit vectors, especially FMAs), and assumes (generally) that newer instruction sets are better. This may be addressed in the future, but is far down on the priority list, since newer Intel CPUs and all AMD CPUs with AVX512 are much less affected. If you're looking for a simple workaround for this, you can either adjust the requirements for 512-bit operations in the source code, or force all AVX512 flags off in `FeatureSet.h` (but the second option will also restrict your access to instructions available in AVX512 that operate on 256 and 128 bit vectors)

# Implementation search order
The current order for implementation search is given below. Upper items have higher priority and considered first. Note that if instruction set appears here, it doesn't mean that is is implemented or even used at all in the library. Instead, this is a general list for currently implemented and possible future operations.

1. AVX512 (extensions order below)
1. AVX2
1. FMA3
1. F16C
1. AVX
1. AES
1. SSE4.2
1. SSE4.1
1. SSE4A
1. SSSE3
1. SSE3
1. SSE2
1. SSE
1. MMX

AVX512 priority:
1. AVX512 BMM
1. AVX512 FP16
1. AVX512 VP2INTERSECT
1. AVX512 BF16
1. AVX512 VAES
1. AVX512 GFNI
1. AVX512 VPCLMULQDQ
1. AVX512 VNNI
1. AVX512 BITALG
1. AVX512 VPOPCNTDQ
1. AVX512 VBMI2
1. AVX512 VBMI
1. AVX512 IFMA
1. AVX512 BW
1. AVX512 DQ
1. AVX512 VL
1. AVX512 CD
1. AVX512 F

# Noisy scalars mode
Defining the `AVXXY_NOISY_SCALAR` macro will dump all calls to scalar implementations to the console. This mode is usually not useful to the users of the library, and is mostly used during library's debugging. You can, however, enable it to inspect possible performance downfalls in case you're sure that no scalar implementations should be used by your code.