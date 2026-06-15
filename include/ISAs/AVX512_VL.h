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
			template<internals::FeatureSet FS>
			struct AVX512VL
			{
				template<typename S, size_t N>
					requires (FS.has(AVX512_F) && any_i64<S>)
				static SIMD_Vector<S, N> eval(op_abs, const SIMD_Vector<S, N>& a)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (ymm_sized<T>) return _mm256_abs_epi64(a);
					else if constexpr (xmm_sized<T>) return _mm_abs_epi64(a);
					else static_assert(always_false_v<S>);
				}

				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_mask_mov, const SIMD_Vector<S, N>& ifBitClear, const SIMD_BitMask<N>& mask, const SIMD_Vector<S, N>& ifBitSet)
					requires (sizeof(S) < 4 && FS.has(AVX512_BW) && (concepts::xmm_sized<SIMD_Vector<S, N>> || concepts::ymm_sized<SIMD_Vector<S, N>>))
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (ymm_sized<T> && any_i16<S>) return _mm256_mask_mov_epi16(ifBitClear, mask, ifBitSet);
					else if constexpr (ymm_sized<T> && any_i8<S>) return _mm256_mask_mov_epi8(ifBitClear, mask, ifBitSet);
					else if constexpr (xmm_sized<T> && any_i16<S>) return _mm_mask_mov_epi16(ifBitClear, mask, ifBitSet);
					else if constexpr (xmm_sized<T> && any_i8<S>) return _mm_mask_mov_epi8(ifBitClear, mask, ifBitSet);
					else static_assert(always_false_v<S>);
				}
			};
		}
	}
}