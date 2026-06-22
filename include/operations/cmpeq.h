#pragma once
#include "shared.h"
namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		struct op_cmpeq : OperationBase
		{
			template<typename S, size_t N>
			static mask_t<S, N> run(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (FS.has(AVX512_BW))
				{
					if constexpr (zmm_sized<T>)
					{
						if constexpr (is_i8<S>) return _mm512_cmpeq_epi8_mask(a, b);
						else if constexpr (is_u8<S>) return _mm512_cmpeq_epu8_mask(a, b);
						else if constexpr (is_i16<S>) return _mm512_cmpeq_epi16_mask(a, b);
						else if constexpr (is_u16<S>) return _mm512_cmpeq_epu16_mask(a, b);
					}
					else if constexpr (FS.has(AVX512_VL) && ymm_sized<T>)
					{

					}
				}
				scream();
				typename SIMD_Vector<S, N>::MaskT ret = 0;
				for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] == b[i]);
				return ret;
			}
		};
	}
}