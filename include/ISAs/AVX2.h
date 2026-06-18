#pragma once
#include "../namespace.h"
#include "../tags.h"
#include "../SIMD_BitMask.h"
#include "../SIMD_Vector.h"
#include "../FeatureSet.h"
namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		namespace ISA
		{
			using namespace concepts;
			using namespace utils;
			template<internals::FeatureSet FS>
			struct AVX2
			{
				template<typename S, size_t N>
				requires (any_small_int<S> && sizeof(SIMD_Vector<S,N>) >= 17) //|| (FS.has(SSSE3) && any_i16<S>))
				static SIMD_BitMask<N> eval(op_vec2mask, const SIMD_Vector<S, N>& a)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { vec2mask(a.lo()),vec2mask(a.hi()) };
					else if constexpr (ymm_sized<T> && any_i8<S>) return _mm256_movemask_epi8(a);
					else if constexpr (FS.has(SSSE3) && ymm_sized<T> && any_i16<S>)
					{
						//AVX2 has no movemask_epi16 intrinsic, so we need to fall back to older shuffle+movemask
						//only care about upper bytes of each 16-bit word. 
						//concentrate upper bytes of each word into lower (x) or upper (y) half of 128-bit vector
						//then merge them and return 16 bit mask
						//cvt to 32 bits + domain cross + movemask_ps may be faster if domain penalties don't apply (Skylake and newer?)
						__m128i x = _mm_shuffle_epi8(a.lo(), _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1));
						__m128i y = _mm_shuffle_epi8(a.hi(), _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 1, 3, 5, 7, 9, 11, 13, 15));
						return _mm_movemask_epi8(_mm_or_si128(x, y));
					}
					else static_assert(always_false_v<T>);
				}

				template<typename S, size_t N>
					requires (any_int<S> && sizeof(SIMD_Vector<S, N>) >= 17)
				static SIMD_Vector<S,N> eval(op_mask2vec, const SIMD_BitMask<N>& a)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { mask2vec<S>(a.lo()),mask2vec<S>(a.hi()) };
					else if constexpr (ymm_sized<T> && any_i64<S>)
					{
						__m256i broadcasted = _mm256_set1_epi64x(a);
						__m256i x = _mm256_andnot_si256(broadcasted, _mm256_setr_epi64x(1, 2, 4, 8));
						return _mm256_cmpeq_epi64(x, _mm256_set1_epi64x(0));
					}
					else if constexpr (ymm_sized<T> && any_i32<S>)
					{
						__m256i broadcasted = _mm256_set1_epi32(a);
						__m256i x = _mm256_andnot_si256(broadcasted, _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128));
						return _mm256_cmpeq_epi32(x, _mm256_set1_epi32(0));
					}
					else if constexpr (ymm_sized<T> && any_i16<S>)
					{
						__m256i broadcasted = _mm256_set1_epi16(a);
						__m256i x = _mm256_andnot_si256(broadcasted, _mm256_setr_epi16(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768));
						return _mm256_cmpeq_epi16(x, _mm256_set1_epi16(0));
					}
					else if constexpr (ymm_sized<T> && any_i8<S>) //too much bits in mask (32 bits don't fit into 8 bits of elements). Thus, split the mask, deposit it into 16 bit elements and then combine
					{
						int16_t masklo = a.lo();
						int16_t maskhi = a.hi();
						static_assert(N % 2 == 0);
						auto vlo = mask2vec<uint16_t, N/2>(masklo);
						auto vhi = mask2vec<uint16_t, N/2>(maskhi) << 8;
						return (vlo & 0xFF) | vhi;
						//return _mm256_blendv_epi8(vlo,vhi, _mm256_setr_epi8())
					}
					else static_assert(always_false_v<T>);
				}

				//Lookup table that maps 8-bit compress mask to 8-element permutexvar index register
				//It it compressed to only take up 1 byte per index, thus, it needs to be expanded after loading at the call site (from int8_t's to your required signed type)
				//Negative indices (when treated as int8_t) stored here means that value is masked out and should be passed through from other register.
				//It is uncertain if 4 bit LUT would be better. 2 KiB size is decently large, but greatly simplifies the compress emulation for 8-element vectors, replacing it just with 1 mask extraction + lookup + cvt + cross-lane permute
				static const std::array<uint64_t, 256> compress_to_permx_lut8 = []() {
					std::array<uint64_t, 256> ret;
					for (int i = 0; i < 256; ++i)
					{
						uint64_t val = 0xFFFFFFFFFFFFFFFF;
						int pivot = 0;
						for (int j = 0; j < 8; ++j)
						{
							if (i & (1 << j))
							{
								val &= ~(uint64_t(0xFF) << (8 * pivot));
								val |= uint64_t(j) << (8 * (pivot++));
							}
						}
						ret[i] = val;
					}
					return ret;
					}();
			};
		}
	}
}