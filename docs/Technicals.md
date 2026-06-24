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