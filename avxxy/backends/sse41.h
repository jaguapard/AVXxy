#pragma once
#include "../vector.h"
#include "sse2.h"

namespace AVXXY_NAMESPACE
{
	namespace backends
	{
		struct SSE41 : SSE2
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> mul(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 16) return { mul(a.lo(), b.lo()), mul(a.hi(), b.hi()) };
				else if constexpr (xmm_sized<T>)
				{
					if constexpr (is_f64<S>) return _mm_mul_pd(a, b);
					else if constexpr (is_f32<S>) return _mm_mul_ps(a, b);
					else if constexpr (any_i32<S>) return _mm_mullo_epi32(a, b);
					else if constexpr (any_i16<S>) return _mm_mullo_epi16(a, b);
					else return Scalar::mul(a, b);
				}
			}

			template<typename S, size_t N>
				requires (std::is_floating_point_v<S>)
			static SIMD_Vector<S, N> floor(const SIMD_Vector<S, N>& a)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 16) return { floor(a.lo()), floor(a.hi()) };
				else if constexpr (std::is_same_v<S, float>) return _mm_floor_ps(a);
				else return _mm_floor_pd(a);
			}
			template<typename S, size_t N>
				requires (std::is_floating_point_v<S>)
			static SIMD_Vector<S, N> ceil(const SIMD_Vector<S, N>& a)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 16) return { ceil(a.lo()), ceil(a.hi()) };
				else if constexpr (std::is_same_v<S, float>) return _mm_ceil_ps(a);
				else return _mm_ceil_pd(a);
			}

			template<typename S, size_t N>
			static SIMD_Vector<S, N> abs(const SIMD_Vector<S, N>& a)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (std::is_unsigned_v<S>) return a;
				else if constexpr (sizeof(T) > 16) return { abs(a.lo()), abs(a.hi()) };
				else if constexpr (is_f32<S>) return _mm_andnot_ps(_mm_set1_ps(-0.0f), a);
				else if constexpr (is_f64<S>) return _mm_andnot_pd(_mm_set1_pd(-0.0), a);
				else if constexpr (is_i32<S>) return _mm_abs_epi32(a);
				else if constexpr (is_i16<S>) return _mm_abs_epi16(a);
				else if constexpr (is_i8<S>) return _mm_abs_epi8(a);
				else return Scalar::abs(a);
			}

			template<typename S, size_t N>
			static SIMD_Vector<S, N> min(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 16) return { min(a.lo(), b.lo()), min(a.hi(), b.hi()) };
				else if constexpr (is_f64<S>) return _mm_min_pd(a, b);
				else if constexpr (is_f32<S>) return _mm_min_ps(a, b);
				else if constexpr (is_i32<S>) return _mm_min_epi32(a, b);
				else if constexpr (is_u32<S>) return _mm_min_epu32(a, b);
				else if constexpr (is_i16<S>) return _mm_min_epi16(a, b);
				else if constexpr (is_u16<S>) return _mm_min_epu16(a, b);
				else if constexpr (is_i8<S>) return _mm_min_epi8(a, b);
				else if constexpr (is_u8<S>) return _mm_min_epu8(a, b);
				else return Scalar::min(a, b);
			}
			template<typename S, size_t N>
			static SIMD_Vector<S, N> max(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 16) return { max(a.lo(), b.lo()), max(a.hi(), b.hi()) };
				else if constexpr (is_f64<S>) return _mm_max_pd(a, b);
				else if constexpr (is_f32<S>) return _mm_max_ps(a, b);
				else if constexpr (is_i32<S>) return _mm_max_epi32(a, b);
				else if constexpr (is_u32<S>) return _mm_max_epu32(a, b);
				else if constexpr (is_i16<S>) return _mm_max_epi16(a, b);
				else if constexpr (is_u16<S>) return _mm_max_epu16(a, b);
				else if constexpr (is_i8<S>) return _mm_max_epi8(a, b);
				else if constexpr (is_u8<S>) return _mm_max_epu8(a, b);
				else return Scalar::max(a, b);
			}
		};
	}
}
