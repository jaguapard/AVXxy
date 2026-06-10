#pragma once
#include "../vector.h"
#include "scalar.h"
#include "../meta_ops.h"

namespace AVXXY_NAMESPACE
{
	namespace backends
	{
		struct SSE2
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> add(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (xmm_sized<T>)
				{
					if constexpr (is_f64<S>) return _mm_add_pd(a, b);
					else if constexpr (is_f32<S>) return _mm_add_ps(a, b);
					else if constexpr (any_i64<S>) return _mm_add_epi64(a, b);
					else if constexpr (any_i32<S>) return _mm_add_epi32(a, b);
					else if constexpr (any_i16<S>) return _mm_add_epi16(a, b);
					else if constexpr (any_i8<S>) return _mm_add_epi8(a, b);
				}
				else return { add(a.lo(), b.lo()), add(a.hi(), b.hi()) };
			}
		};
	}
}