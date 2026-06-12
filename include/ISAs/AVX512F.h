#pragma once
#pragma once
#include "../namespace.h"
#include "../tags.h"
#include "../SIMD_BitMask.h"
#include "../SIMD_Vector.h"
namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		namespace ISA
		{
			struct AVX512F
			{
				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_mask_mov, const SIMD_Vector<S, N>& ifBitClear, const SIMD_BitMask<N>& mask, const SIMD_Vector<S, N>& ifBitSet)
					requires (sizeof(S) == 4 || sizeof(S) == 8)
				{
					using namespace concepts;
					if constexpr (is_f64<S>) return _mm512_mask_mov_pd(ifBitClear, mask, ifBitSet);
					else if constexpr (is_f64<S>) return _mm512_mask_mov_ps(ifBitClear, mask, ifBitSet);
					else if constexpr (any_i64<S>) return _mm512_mask_mov_epi64(ifBitClear, mask, ifBitSet);
					else if constexpr (any_i32<S>) return _mm512_mask_mov_epi32(ifBitClear, mask, ifBitSet);
				}
			};
		}
	}
};