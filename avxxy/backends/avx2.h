#pragma once
#include "../vector.h"
#include "sse41.h"

namespace AVXXY_NAMESPACE
{
	namespace backends
	{
		struct AVX2 : SSE41
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> add(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 32) return { add(a.lo(), b.lo()), add(a.hi(), b.hi()) };
				else if constexpr (ymm_sized<T>)
				{
					if constexpr (is_f64<S>) return _mm256_add_pd(a, b);
					else if constexpr (is_f32<S>) return _mm256_add_ps(a, b);
					else if constexpr (any_i64<S>) return _mm256_add_epi64(a, b);
					else if constexpr (any_i32<S>) return _mm256_add_epi32(a, b);
					else if constexpr (any_i16<S>) return _mm256_add_epi16(a, b);
					else if constexpr (any_i8<S>) return _mm256_add_epi8(a, b);
				}
				else return SSE41::add(a, b);
			}

			template<typename S, size_t N>
			static SIMD_Vector<S, N> sub(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 32) return { sub(a.lo(), b.lo()), sub(a.hi(), b.hi()) };
				else if constexpr (ymm_sized<T>)
				{
					if constexpr (is_f64<S>) return _mm256_sub_pd(a, b);
					else if constexpr (is_f32<S>) return _mm256_sub_ps(a, b);
					else if constexpr (any_i64<S>) return _mm256_sub_epi64(a, b);
					else if constexpr (any_i32<S>) return _mm256_sub_epi32(a, b);
					else if constexpr (any_i16<S>) return _mm256_sub_epi16(a, b);
					else if constexpr (any_i8<S>) return _mm256_sub_epi8(a, b);
				}
				else return SSE41::sub(a, b);
			}

			template<typename S, size_t N>
			static SIMD_Vector<S, N> mul(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 32) return { mul(a.lo(), b.lo()), mul(a.hi(), b.hi()) };
				else if constexpr (ymm_sized<T>)
				{
					if constexpr (is_f64<S>) return _mm256_mul_pd(a, b);
					else if constexpr (is_f32<S>) return _mm256_mul_ps(a, b);
					else if constexpr (any_i32<S>) return _mm256_mullo_epi32(a, b);
					else if constexpr (any_i16<S>) return _mm256_mullo_epi16(a, b);
					else return Scalar::mul(a, b);
				}
				else return SSE41::mul(a, b);
			}

			template<typename S, size_t N>
			static SIMD_Vector<S, N> div(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 32) return { div(a.lo(), b.lo()), div(a.hi(), b.hi()) };
				else if constexpr (ymm_sized<T>)
				{
					if constexpr (is_f64<S>) return _mm256_div_pd(a, b);
					else if constexpr (is_f32<S>) return _mm256_div_ps(a, b);
					else return Scalar::div(a, b);
				}
				else return SSE41::div(a, b);
			}

			template<typename S, size_t N>
			static SIMD_Vector<S, N> logic_or(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 32) return { logic_or(a.lo(), b.lo()), logic_or(a.hi(), b.hi()) };
				else if constexpr (concepts::ymm_sized<T>) return T(_mm256_or_si256(__m256i(a), __m256i(b)));
				else return SSE41::logic_or(a, b);
			}
			template<typename S, size_t N>
			static SIMD_Vector<S, N> logic_and(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 32) return { logic_and(a.lo(), b.lo()), logic_and(a.hi(), b.hi()) };
				else if constexpr (concepts::ymm_sized<T>) return T(_mm256_and_si256(__m256i(a), __m256i(b)));
				else return SSE41::logic_and(a, b);
			}
			template<typename S, size_t N>
			static SIMD_Vector<S, N> logic_xor(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 32) return { logic_xor(a.lo(), b.lo()), logic_xor(a.hi(), b.hi()) };
				else if constexpr (concepts::ymm_sized<T>) return T(_mm256_xor_si256(__m256i(a), __m256i(b)));
				else return SSE41::logic_xor(a, b);
			}
			template<typename S, size_t N>
			static SIMD_Vector<S, N> logic_not(const SIMD_Vector<S, N>& a)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 32) return { logic_not(a.lo()), logic_not(a.hi()) };
				else if constexpr (concepts::ymm_sized<T>) return T(_mm256_xor_si256(__m256i(a), _mm256_set1_epi32(-1)));
				else return SSE41::logic_not(a);
			}

			template<typename S, size_t N>
			static SIMD_Vector<float, N> sqrtf(const SIMD_Vector<S, N>& a)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (std::is_same_v<S, float> && sizeof(T) > 32) return { sqrtf(a.lo()), sqrtf(a.hi()) };
				else if constexpr (std::is_same_v<S, float> && concepts::ymm_sized<T>) return _mm256_sqrt_ps(a);
				else return SSE41::sqrtf(a);
			}
			template<typename S, size_t N>
			static SIMD_Vector<double, N> sqrtd(const SIMD_Vector<S, N>& a)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (std::is_same_v<S, double> && sizeof(T) > 32) return { sqrtd(a.lo()), sqrtd(a.hi()) };
				else if constexpr (std::is_same_v<S, double> && concepts::ymm_sized<T>) return _mm256_sqrt_pd(a);
				else return SSE41::sqrtd(a);
			}

			template<typename S, size_t N>
				requires (std::is_floating_point_v<S>)
			static SIMD_Vector<S, N> floor(const SIMD_Vector<S, N>& a)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 32) return { floor(a.lo()), floor(a.hi()) };
				else if constexpr (concepts::ymm_sized<T> && std::is_same_v<S, float>) return _mm256_floor_ps(a);
				else if constexpr (concepts::ymm_sized<T>) return _mm256_floor_pd(a);
				else return SSE41::floor(a);
			}
			template<typename S, size_t N>
				requires (std::is_floating_point_v<S>)
			static SIMD_Vector<S, N> ceil(const SIMD_Vector<S, N>& a)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 32) return { ceil(a.lo()), ceil(a.hi()) };
				else if constexpr (concepts::ymm_sized<T> && std::is_same_v<S, float>) return _mm256_ceil_ps(a);
				else if constexpr (concepts::ymm_sized<T>) return _mm256_ceil_pd(a);
				else return SSE41::ceil(a);
			}

			template<typename S, size_t N>
			static SIMD_Vector<S, N> unpacklo(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 32) return { unpacklo(a.lo(), b.lo()), unpacklo(a.hi(), b.hi()) };
				else if constexpr (ymm_sized<T>)
				{
					if constexpr (is_f64<S>) return _mm256_unpacklo_pd(a, b);
					else if constexpr (is_f32<S>) return _mm256_unpacklo_ps(a, b);
					else if constexpr (any_i64<S>) return _mm256_unpacklo_epi64(a, b);
					else if constexpr (any_i32<S>) return _mm256_unpacklo_epi32(a, b);
					else if constexpr (any_i16<S>) return _mm256_unpacklo_epi16(a, b);
					else if constexpr (any_i8<S>) return _mm256_unpacklo_epi8(a, b);
				}
				else return SSE41::unpacklo(a, b);
			}
			template<typename S, size_t N>
			static SIMD_Vector<S, N> unpackhi(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 32) return { unpackhi(a.lo(), b.lo()), unpackhi(a.hi(), b.hi()) };
				else if constexpr (ymm_sized<T>)
				{
					if constexpr (is_f64<S>) return _mm256_unpackhi_pd(a, b);
					else if constexpr (is_f32<S>) return _mm256_unpackhi_ps(a, b);
					else if constexpr (any_i64<S>) return _mm256_unpackhi_epi64(a, b);
					else if constexpr (any_i32<S>) return _mm256_unpackhi_epi32(a, b);
					else if constexpr (any_i16<S>) return _mm256_unpackhi_epi16(a, b);
					else if constexpr (any_i8<S>) return _mm256_unpackhi_epi8(a, b);
				}
				else return SSE41::unpackhi(a, b);
			}

			template<typename S, size_t N>
			static SIMD_Vector<S, N> abs(const SIMD_Vector<S, N>& a)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (std::is_unsigned_v<S>) return a;
				else if constexpr (sizeof(T) > 32) return { abs(a.lo()), abs(a.hi()) };
				else if constexpr (ymm_sized<T> && is_f32<S>) return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a);
				else if constexpr (ymm_sized<T> && is_f64<S>) return _mm256_andnot_pd(_mm256_set1_pd(-0.0), a);
				else if constexpr (ymm_sized<T> && is_i32<S>) return _mm256_abs_epi32(a);
				else if constexpr (ymm_sized<T> && is_i16<S>) return _mm256_abs_epi16(a);
				else if constexpr (ymm_sized<T> && is_i8<S>) return _mm256_abs_epi8(a);
				else return SSE41::abs(a);
			}

			template<typename S, size_t N>
			static SIMD_Vector<S, N> min(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 32) return { min(a.lo(), b.lo()), min(a.hi(), b.hi()) };
				else if constexpr (ymm_sized<T>)
				{
					if constexpr (is_f64<S>) return _mm256_min_pd(a, b);
					else if constexpr (is_f32<S>) return _mm256_min_ps(a, b);
					else if constexpr (is_i32<S>) return _mm256_min_epi32(a, b);
					else if constexpr (is_u32<S>) return _mm256_min_epu32(a, b);
					else if constexpr (is_i16<S>) return _mm256_min_epi16(a, b);
					else if constexpr (is_u16<S>) return _mm256_min_epu16(a, b);
					else if constexpr (is_i8<S>) return _mm256_min_epi8(a, b);
					else if constexpr (is_u8<S>) return _mm256_min_epu8(a, b);
					else return Scalar::min(a, b);
				}
				else return SSE41::min(a, b);
			}
			template<typename S, size_t N>
			static SIMD_Vector<S, N> max(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 32) return { max(a.lo(), b.lo()), max(a.hi(), b.hi()) };
				else if constexpr (ymm_sized<T>)
				{
					if constexpr (is_f64<S>) return _mm256_max_pd(a, b);
					else if constexpr (is_f32<S>) return _mm256_max_ps(a, b);
					else if constexpr (is_i32<S>) return _mm256_max_epi32(a, b);
					else if constexpr (is_u32<S>) return _mm256_max_epu32(a, b);
					else if constexpr (is_i16<S>) return _mm256_max_epi16(a, b);
					else if constexpr (is_u16<S>) return _mm256_max_epu16(a, b);
					else if constexpr (is_i8<S>) return _mm256_max_epi8(a, b);
					else if constexpr (is_u8<S>) return _mm256_max_epu8(a, b);
					else return Scalar::max(a, b);
				}
				else return SSE41::max(a, b);
			}
		};
	}
}
