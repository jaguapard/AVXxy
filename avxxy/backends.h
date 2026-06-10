#pragma once
#include "namespace.h"
#include "concepts.h"
#include "vector.h"

namespace AVXXY_NAMESPACE
{
	namespace backends
	{
		struct Scalar
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> add(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i] + b[i];
				return ret;
			}
		};

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
		struct SSE41
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> add(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				return SSE2::add(a, b);
			}
		};
		struct AVX2
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
		};

		struct AVX512
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> add(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return { add(a.lo(), b.lo()), add(a.hi(), b.hi()) };
				else if constexpr (zmm_sized<T>)
				{
					if constexpr (is_f64<S>) return _mm512_add_pd(a, b);
					else if constexpr (is_f32<S>) return _mm512_add_ps(a, b);
					else if constexpr (any_i64<S>) return _mm512_add_epi64(a, b);
					else if constexpr (any_i32<S>) return _mm512_add_epi32(a, b);
					else if constexpr (any_i16<S>) return _mm512_add_epi16(a, b);
					else if constexpr (any_i8<S>) return _mm512_add_epi8(a, b);
				}
				else return AVX2::add(a, b);
			}
		};

		
		
		

		typedef AVX512 current;
	}
}