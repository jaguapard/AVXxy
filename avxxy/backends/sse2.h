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
				if constexpr (sizeof(T) > 16) return { add(a.lo(), b.lo()), add(a.hi(), b.hi()) };
				else if constexpr (xmm_sized<T>)
				{
					if constexpr (is_f64<S>) return _mm_add_pd(a, b);
					else if constexpr (is_f32<S>) return _mm_add_ps(a, b);
					else if constexpr (any_i64<S>) return _mm_add_epi64(a, b);
					else if constexpr (any_i32<S>) return _mm_add_epi32(a, b);
					else if constexpr (any_i16<S>) return _mm_add_epi16(a, b);
					else if constexpr (any_i8<S>) return _mm_add_epi8(a, b);
				}
			}

			template<typename S, size_t N>
			static SIMD_Vector<S, N> sub(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 16) return { sub(a.lo(), b.lo()), sub(a.hi(), b.hi()) };
				else if constexpr (xmm_sized<T>)
				{
					if constexpr (is_f64<S>) return _mm_sub_pd(a, b);
					else if constexpr (is_f32<S>) return _mm_sub_ps(a, b);
					else if constexpr (any_i64<S>) return _mm_sub_epi64(a, b);
					else if constexpr (any_i32<S>) return _mm_sub_epi32(a, b);
					else if constexpr (any_i16<S>) return _mm_sub_epi16(a, b);
					else if constexpr (any_i8<S>) return _mm_sub_epi8(a, b);
				}
			}

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
					else if constexpr (any_i16<S>) return _mm_mullo_epi16(a, b);
					else return Scalar::mul(a, b);
				}
			}

			template<typename S, size_t N>
			static SIMD_Vector<S, N> div(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 16) return { div(a.lo(), b.lo()), div(a.hi(), b.hi()) };
				else if constexpr (xmm_sized<T>)
				{
					if constexpr (is_f64<S>) return _mm_div_pd(a, b);
					else if constexpr (is_f32<S>) return _mm_div_ps(a, b);
					else return Scalar::div(a, b);
				}
			}

			template<typename S, size_t N>
			static SIMD_Vector<S, N> mod(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { return Scalar::mod(a, b); }

			template<typename S, size_t N>
			static SIMD_Vector<S, N> logic_or(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 16) return { logic_or(a.lo(), b.lo()), logic_or(a.hi(), b.hi()) };
				else return T(_mm_or_si128(__m128i(a), __m128i(b)));
			}
			template<typename S, size_t N>
			static SIMD_Vector<S, N> logic_and(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 16) return { logic_and(a.lo(), b.lo()), logic_and(a.hi(), b.hi()) };
				else return T(_mm_and_si128(__m128i(a), __m128i(b)));
			}
			template<typename S, size_t N>
			static SIMD_Vector<S, N> logic_xor(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 16) return { logic_xor(a.lo(), b.lo()), logic_xor(a.hi(), b.hi()) };
				else return T(_mm_xor_si128(__m128i(a), __m128i(b)));
			}
			template<typename S, size_t N>
			static SIMD_Vector<S, N> logic_not(const SIMD_Vector<S, N>& a)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 16) return { logic_not(a.lo()), logic_not(a.hi()) };
				else return T(_mm_xor_si128(__m128i(a), _mm_set1_epi32(-1)));
			}

			template<typename S, size_t N, typename I>
				requires (concepts::any_int<S> && concepts::any_int<I>)
			static SIMD_Vector<S, N> shift_left(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b) { return Scalar::shift_left(a, b); }
			template<typename S, size_t N, typename I>
				requires (concepts::any_int<S> && concepts::any_int<I>)
			static SIMD_Vector<S, N> shift_right(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b) { return Scalar::shift_right(a, b); }

			template<typename S, size_t N>
			static SIMD_Vector<float, N> sqrtf(const SIMD_Vector<S, N>& a)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (std::is_same_v<S, float> && sizeof(T) > 16) return { sqrtf(a.lo()), sqrtf(a.hi()) };
				else if constexpr (std::is_same_v<S, float> && xmm_sized<T>) return _mm_sqrt_ps(a);
				else return Scalar::sqrtf(a);
			}
			template<typename S, size_t N>
			static SIMD_Vector<double, N> sqrtd(const SIMD_Vector<S, N>& a)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (std::is_same_v<S, double> && sizeof(T) > 16) return { sqrtd(a.lo()), sqrtd(a.hi()) };
				else if constexpr (std::is_same_v<S, double> && xmm_sized<T>) return _mm_sqrt_pd(a);
				else return Scalar::sqrtd(a);
			}

			template<typename S, size_t N, typename I>
				requires (concepts::any_int<I>)
			static SIMD_Vector<S, N> permx(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind) { return Scalar::permx(a, ind); }
			template<typename S, size_t N, typename I>
				requires (concepts::any_int<I>)
			static SIMD_Vector<S, N> permx2(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind, const SIMD_Vector<S, N>& b) { return Scalar::permx2(a, ind, b); }

			template<typename S, size_t N>
				requires (std::is_floating_point_v<S>)
			static SIMD_Vector<S, N> floor(const SIMD_Vector<S, N>& a) { return Scalar::floor(a); }
			template<typename S, size_t N>
				requires (std::is_floating_point_v<S>)
			static SIMD_Vector<S, N> ceil(const SIMD_Vector<S, N>& a) { return Scalar::ceil(a); }

			template<typename S, size_t N>
			static SIMD_Vector<S, N> load(const void* p, const SIMD_Mask<N>& mask = SIMD_Mask<N>::AllOnes, const SIMD_Vector<S, N>& src = 0)
			{
				using T = SIMD_Vector<S, N>;
				if (typename SIMD_Mask<N>::UintT(mask) != SIMD_Mask<N>::AllOnes) return Scalar::load<S, N>(p, mask, src);
				if constexpr (sizeof(T) > 16) return { load<S>(p, mask.lo(), src.lo()), load<S>(static_cast<const S*>(p) + N / 2, mask.hi(), src.hi()) };
				else { T ret; memcpy(&ret, p, sizeof(ret)); return ret; }
			}
			template<typename S, size_t N>
			static void store(SIMD_Vector<S, N> vec, void* p, const SIMD_Mask<N>& mask = SIMD_Mask<N>::AllOnes)
			{
				if (typename SIMD_Mask<N>::UintT(mask) != SIMD_Mask<N>::AllOnes) return Scalar::store(vec, p, mask);
				memcpy(p, &vec, sizeof(vec));
			}
			template<typename S, size_t N, size_t Scale, typename I>
				requires (concepts::any_int<I>)
			static SIMD_Vector<S, N> gather(const void* base, const SIMD_Vector<I, N>& ind, const SIMD_Mask<N>& mask = SIMD_Mask<N>::AllOnes, const SIMD_Vector<S, N>& src = 0) { return Scalar::gather<S, N, Scale>(base, ind, mask, src); }
			template<typename S, size_t N, size_t Scale, typename I>
				requires (concepts::any_int<I>)
			static void scatter(const SIMD_Vector<S, N>& v, void* base, const SIMD_Vector<I, N>& ind, const SIMD_Mask<N>& mask = SIMD_Mask<N>::AllOnes) { Scalar::scatter<S, N, Scale>(v, base, ind, mask); }

			template<typename S, size_t N>
			static SIMD_Vector<S, N> compress(const SIMD_Mask<N>& mask, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& src = 0) { return Scalar::compress(mask, a, src); }

			template<typename S, size_t N>
			static SIMD_Vector<S, N> mask_mov(const SIMD_Vector<S, N>& ifBitClear, const SIMD_Mask<N>& mask, const SIMD_Vector<S, N>& ifBitSet) { return Scalar::mask_mov(ifBitClear, mask, ifBitSet); }

			template<typename S, size_t N>
			static SIMD_Vector<S, N> unpacklo(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 16) return { unpacklo(a.lo(), b.lo()), unpacklo(a.hi(), b.hi()) };
				else if constexpr (is_f64<S>) return _mm_unpacklo_pd(a, b);
				else if constexpr (is_f32<S>) return _mm_unpacklo_ps(a, b);
				else if constexpr (any_i64<S>) return _mm_unpacklo_epi64(a, b);
				else if constexpr (any_i32<S>) return _mm_unpacklo_epi32(a, b);
				else if constexpr (any_i16<S>) return _mm_unpacklo_epi16(a, b);
				else if constexpr (any_i8<S>) return _mm_unpacklo_epi8(a, b);
			}
			template<typename S, size_t N>
			static SIMD_Vector<S, N> unpackhi(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 16) return { unpackhi(a.lo(), b.lo()), unpackhi(a.hi(), b.hi()) };
				else if constexpr (is_f64<S>) return _mm_unpackhi_pd(a, b);
				else if constexpr (is_f32<S>) return _mm_unpackhi_ps(a, b);
				else if constexpr (any_i64<S>) return _mm_unpackhi_epi64(a, b);
				else if constexpr (any_i32<S>) return _mm_unpackhi_epi32(a, b);
				else if constexpr (any_i16<S>) return _mm_unpackhi_epi16(a, b);
				else if constexpr (any_i8<S>) return _mm_unpackhi_epi8(a, b);
			}

			template<typename S, size_t N> static SIMD_Mask<N> cmp_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { return Scalar::cmp_equal(a, b); }
			template<typename S, size_t N> static SIMD_Mask<N> cmp_not_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { return Scalar::cmp_not_equal(a, b); }
			template<typename S, size_t N> static SIMD_Mask<N> cmp_less(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { return Scalar::cmp_less(a, b); }
			template<typename S, size_t N> static SIMD_Mask<N> cmp_less_or_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { return Scalar::cmp_less_or_equal(a, b); }
			template<typename S, size_t N> static SIMD_Mask<N> cmp_greater(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { return Scalar::cmp_greater(a, b); }
			template<typename S, size_t N> static SIMD_Mask<N> cmp_greater_or_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { return Scalar::cmp_greater_or_equal(a, b); }

			template<typename S, size_t N> static SIMD_Vector<S, N> abs(const SIMD_Vector<S, N>& a) { return Scalar::abs(a); }
			template<typename S, size_t N>
			static SIMD_Vector<S, N> min(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 16) return { min(a.lo(), b.lo()), min(a.hi(), b.hi()) };
				else if constexpr (is_f64<S>) return _mm_min_pd(a, b);
				else if constexpr (is_f32<S>) return _mm_min_ps(a, b);
				else if constexpr (is_u8<S>) return _mm_min_epu8(a, b);
				else if constexpr (is_i16<S>) return _mm_min_epi16(a, b);
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
				else if constexpr (is_u8<S>) return _mm_max_epu8(a, b);
				else if constexpr (is_i16<S>) return _mm_max_epi16(a, b);
				else return Scalar::max(a, b);
			}

			template <size_t N> static SIMD_Vector<uint16_t, N> vec_ps2ph(const SIMD_Vector<float, N>& a) { return Scalar::vec_ps2ph(a); }
			template <size_t N> static SIMD_Vector<float, N> vec_ph2ps(const SIMD_Vector<uint16_t, N>& a) { return Scalar::vec_ph2ps(a); }
		};
	}
}
