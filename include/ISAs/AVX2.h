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
			};
		}
	}
}