#pragma once
#include "../vector.h"
#include "avx2.h"

namespace AVXXY_NAMESPACE
{
	namespace backends
	{
		struct AVX512 : AVX2
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

			template<typename S, size_t N>
			static SIMD_Vector<S, N> sub(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return { sub(a.lo(), b.lo()), sub(a.hi(), b.hi()) };
				else if constexpr (zmm_sized<T>)
				{
					if constexpr (is_f64<S>) return _mm512_sub_pd(a, b);
					else if constexpr (is_f32<S>) return _mm512_sub_ps(a, b);
					else if constexpr (any_i64<S>) return _mm512_sub_epi64(a, b);
					else if constexpr (any_i32<S>) return _mm512_sub_epi32(a, b);
					else if constexpr (any_i16<S>) return _mm512_sub_epi16(a, b);
					else if constexpr (any_i8<S>) return _mm512_sub_epi8(a, b);
				}
				else return AVX2::sub(a, b);
			}

			template<typename S, size_t N>
			static SIMD_Vector<S, N> mul(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return { mul(a.lo(), b.lo()), mul(a.hi(), b.hi()) };
				else if constexpr (zmm_sized<T>)
				{
					if constexpr (is_f64<S>) return _mm512_mul_pd(a, b);
					else if constexpr (is_f32<S>) return _mm512_mul_ps(a, b);
					else if constexpr (any_i32<S>) return _mm512_mullo_epi32(a, b);
					else if constexpr (any_i16<S>) return _mm512_mullo_epi16(a, b);
					else return Scalar::mul(a, b);
				}
				else return AVX2::mul(a, b);
			}

			template<typename S, size_t N>
			static SIMD_Vector<S, N> div(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return { div(a.lo(), b.lo()), div(a.hi(), b.hi()) };
				else if constexpr (zmm_sized<T>)
				{
					if constexpr (is_f64<S>) return _mm512_div_pd(a, b);
					else if constexpr (is_f32<S>) return _mm512_div_ps(a, b);
					else return Scalar::div(a, b);
				}
				else return AVX2::div(a, b);
			}

			template<typename S, size_t N>
			static SIMD_Vector<S, N> logic_or(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return { logic_or(a.lo(), b.lo()), logic_or(a.hi(), b.hi()) };
				else if constexpr (concepts::zmm_sized<T>) return T(_mm512_or_si512(__m512i(a), __m512i(b)));
				else return AVX2::logic_or(a, b);
			}
			template<typename S, size_t N>
			static SIMD_Vector<S, N> logic_and(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return { logic_and(a.lo(), b.lo()), logic_and(a.hi(), b.hi()) };
				else if constexpr (concepts::zmm_sized<T>) return T(_mm512_and_si512(__m512i(a), __m512i(b)));
				else return AVX2::logic_and(a, b);
			}
			template<typename S, size_t N>
			static SIMD_Vector<S, N> logic_xor(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return { logic_xor(a.lo(), b.lo()), logic_xor(a.hi(), b.hi()) };
				else if constexpr (concepts::zmm_sized<T>) return T(_mm512_xor_si512(__m512i(a), __m512i(b)));
				else return AVX2::logic_xor(a, b);
			}
			template<typename S, size_t N>
			static SIMD_Vector<S, N> logic_not(const SIMD_Vector<S, N>& a)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return { logic_not(a.lo()), logic_not(a.hi()) };
				else if constexpr (concepts::zmm_sized<T>) return T(_mm512_xor_si512(__m512i(a), _mm512_set1_epi32(-1)));
				else return AVX2::logic_not(a);
			}

			template<typename S, size_t N>
			static SIMD_Vector<float, N> sqrtf(const SIMD_Vector<S, N>& a)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (std::is_same_v<S, float> && sizeof(T) > 64) return { sqrtf(a.lo()), sqrtf(a.hi()) };
				else if constexpr (std::is_same_v<S, float> && concepts::zmm_sized<T>) return _mm512_sqrt_ps(a);
				else return AVX2::sqrtf(a);
			}
			template<typename S, size_t N>
			static SIMD_Vector<double, N> sqrtd(const SIMD_Vector<S, N>& a)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (std::is_same_v<S, double> && sizeof(T) > 64) return { sqrtd(a.lo()), sqrtd(a.hi()) };
				else if constexpr (std::is_same_v<S, double> && concepts::zmm_sized<T>) return _mm512_sqrt_pd(a);
				else return AVX2::sqrtd(a);
			}

			template<typename S, size_t N>
				requires (std::is_floating_point_v<S>)
			static SIMD_Vector<S, N> floor(const SIMD_Vector<S, N>& a)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return { floor(a.lo()), floor(a.hi()) };
				else if constexpr (concepts::zmm_sized<T> && std::is_same_v<S, float>) return _mm512_floor_ps(a);
				else if constexpr (concepts::zmm_sized<T>) return _mm512_floor_pd(a);
				else return AVX2::floor(a);
			}
			template<typename S, size_t N>
				requires (std::is_floating_point_v<S>)
			static SIMD_Vector<S, N> ceil(const SIMD_Vector<S, N>& a)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return { ceil(a.lo()), ceil(a.hi()) };
				else if constexpr (concepts::zmm_sized<T> && std::is_same_v<S, float>) return _mm512_ceil_ps(a);
				else if constexpr (concepts::zmm_sized<T>) return _mm512_ceil_pd(a);
				else return AVX2::ceil(a);
			}

			template<typename S, size_t N>
			static SIMD_Vector<S, N> compress(const SIMD_Mask<N>& mask, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& src = 0)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return { compress(mask.lo(), a.lo(), src.lo()), compress(mask.hi(), a.hi(), src.hi()) };
				else if constexpr (zmm_sized<T>)
				{
					if constexpr (is_f64<S>) return _mm512_mask_compress_pd(src, mask, a);
					else if constexpr (is_f32<S>) return _mm512_mask_compress_ps(src, mask, a);
					else if constexpr (any_i64<S>) return _mm512_mask_compress_epi64(src, mask, a);
					else if constexpr (any_i32<S>) return _mm512_mask_compress_epi32(src, mask, a);
					else return Scalar::compress(mask, a, src);
				}
				else return AVX2::compress(mask, a, src);
			}

			template<typename S, size_t N>
			static SIMD_Vector<S, N> min(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return { min(a.lo(), b.lo()), min(a.hi(), b.hi()) };
				else if constexpr (zmm_sized<T>)
				{
					if constexpr (is_f64<S>) return _mm512_min_pd(a, b);
					else if constexpr (is_f32<S>) return _mm512_min_ps(a, b);
					else if constexpr (is_i64<S>) return _mm512_min_epi64(a, b);
					else if constexpr (is_u64<S>) return _mm512_min_epu64(a, b);
					else if constexpr (is_i32<S>) return _mm512_min_epi32(a, b);
					else if constexpr (is_u32<S>) return _mm512_min_epu32(a, b);
					else if constexpr (is_i16<S>) return _mm512_min_epi16(a, b);
					else if constexpr (is_u16<S>) return _mm512_min_epu16(a, b);
					else if constexpr (is_i8<S>) return _mm512_min_epi8(a, b);
					else if constexpr (is_u8<S>) return _mm512_min_epu8(a, b);
				}
				else return AVX2::min(a, b);
			}
			template<typename S, size_t N>
			static SIMD_Vector<S, N> max(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace concepts;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return { max(a.lo(), b.lo()), max(a.hi(), b.hi()) };
				else if constexpr (zmm_sized<T>)
				{
					if constexpr (is_f64<S>) return _mm512_max_pd(a, b);
					else if constexpr (is_f32<S>) return _mm512_max_ps(a, b);
					else if constexpr (is_i64<S>) return _mm512_max_epi64(a, b);
					else if constexpr (is_u64<S>) return _mm512_max_epu64(a, b);
					else if constexpr (is_i32<S>) return _mm512_max_epi32(a, b);
					else if constexpr (is_u32<S>) return _mm512_max_epu32(a, b);
					else if constexpr (is_i16<S>) return _mm512_max_epi16(a, b);
					else if constexpr (is_u16<S>) return _mm512_max_epu16(a, b);
					else if constexpr (is_i8<S>) return _mm512_max_epi8(a, b);
					else if constexpr (is_u8<S>) return _mm512_max_epu8(a, b);
				}
				else return AVX2::max(a, b);
			}

			template<typename To, size_t N, typename From>
			static SIMD_Vector<To, N> cvt(const SIMD_Vector<From, N>& a)
			{
				using TV = SIMD_Vector<To, N>;
				using FV = SIMD_Vector<From, N>;
				using namespace concepts;
				constexpr size_t MaxSize = std::max(sizeof(TV), sizeof(FV));

				//converting to self does nothing and can be returned immediately
				if constexpr (std::is_same_v<To, From>) return a;
				//same sized integers reinterpret
				else if constexpr (any_int<From> && any_int<To> && sizeof(From) == sizeof(To)) return vcast<TV>(a);
				//No direct small int -> FP conversions, proxy through 32-bit integers of same signedness
				else if constexpr (sizeof(From) < 4 && any_int<From> && not_int<To>)
				{
					if constexpr (std::is_signed_v<From>) return cvt<To>(cvt<int32_t>(a));
					else if constexpr (std::is_unsigned_v<From>) return cvt<To>(cvt<uint32_t>(a));
				}
				//same for FP -> small int
				else if constexpr (sizeof(To) < 4 && any_int<To> && not_int<From>)
				{
					if constexpr (std::is_signed_v<To>) return cvt<To>(cvt<int32_t>(a));
					else if constexpr (std::is_unsigned_v<To>) return cvt<To>(cvt<uint32_t>(a));
				}
				//split large vectors into halves
				else if constexpr (MaxSize > 64) return { cvt<To>(a.lo()), cvt<To>(a.hi()) };

				//AVX2 has a ton of holes (absent conversions) that will be plugged separately. Thus, all of the conversions are just sitting here
				else if constexpr (zmm_sized<TV> || zmm_sized<FV>)
				{
					//Truncation is REQUIRED for FP -> int
					if constexpr (std::is_same_v<From, double> && std::is_same_v<To, uint64_t>)  return TV(_mm512_cvttpd_epu64(a));
					else if constexpr (std::is_same_v<From, double> && std::is_same_v<To, int64_t>)  return TV(_mm512_cvttpd_epi64(a));
					else if constexpr (std::is_same_v<From, double> && std::is_same_v<To, float>)  return TV(_mm512_cvtpd_ps(a));
					else if constexpr (std::is_same_v<From, double> && std::is_same_v<To, uint32_t>)  return TV(_mm512_cvttpd_epu32(a));
					else if constexpr (std::is_same_v<From, double> && std::is_same_v<To, int32_t>)  return TV(_mm512_cvttpd_epi32(a));

					else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, double>) return TV(_mm512_cvtps_pd(a));
					else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, uint64_t>) return TV(_mm512_cvttps_epu64(a));
					else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, int64_t>) return TV(_mm512_cvttps_epi64(a));
					else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, uint32_t>) return TV(_mm512_cvttps_epu32(a));
					else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, int32_t>) return TV(_mm512_cvttps_epi32(a));
					//TODO: FP16 conversion
					//else if constexpr (std::is_same_v<From, float16_t> && std::is_same_v<To, float>) return TV(_mm512_cvtph_ps(a));

					//only float conversions care about the signedness
					else if constexpr (std::is_same_v<From, uint64_t> && std::is_same_v<To, double>) return TV(_mm512_cvtepu64_pd(a));
					else if constexpr (std::is_same_v<From, uint64_t> && std::is_same_v<To, float>) return TV(_mm512_cvtepu64_ps(a));
					else if constexpr (std::is_same_v<From, int64_t> && std::is_same_v<To, double>) return TV(_mm512_cvtepi64_pd(a));
					else if constexpr (std::is_same_v<From, int64_t> && std::is_same_v<To, float>) return TV(_mm512_cvtepi64_ps(a));
					else if constexpr (std::is_same_v<From, uint32_t> && std::is_same_v<To, double>) return TV(_mm512_cvtepu32_pd(a));
					else if constexpr (std::is_same_v<From, uint32_t> && std::is_same_v<To, float>) return TV(_mm512_cvtepu32_ps(a));
					else if constexpr (std::is_same_v<From, int32_t> && std::is_same_v<To, double>) return TV(_mm512_cvtepi32_pd(a));
					else if constexpr (std::is_same_v<From, int32_t> && std::is_same_v<To, float>) return TV(_mm512_cvtepi32_ps(a));

					//integer conversion zone
					else if constexpr (std::is_integral_v<From> && std::is_integral_v<To>)
					{
						//conversion to smaller ints doesn't care about sign
						if constexpr (sizeof(From) == 8 && sizeof(To) == 4) return TV(_mm512_cvtepi64_epi32(a));
						else if constexpr (sizeof(From) == 8 && sizeof(To) == 2) return TV(_mm512_cvtepi64_epi16(a));
						else if constexpr (sizeof(From) == 8 && sizeof(To) == 1) return TV(_mm512_cvtepi64_epi8(a));
						else if constexpr (sizeof(From) == 4 && sizeof(To) == 2) return TV(_mm512_cvtepi32_epi16(a));
						else if constexpr (sizeof(From) == 4 && sizeof(To) == 1) return TV(_mm512_cvtepi32_epi8(a));
						else if constexpr (sizeof(From) == 2 && sizeof(To) == 1) return TV(_mm512_cvtepi16_epi8(a));

						//conversion to bigger DOES care about sign
						else if constexpr (std::is_same_v<From, uint8_t> && sizeof(To) == 8) return TV(_mm512_cvtepu8_epi64(a));
						else if constexpr (std::is_same_v<From, uint8_t> && sizeof(To) == 4) return TV(_mm512_cvtepu8_epi32(a));
						else if constexpr (std::is_same_v<From, uint8_t> && sizeof(To) == 2) return TV(_mm512_cvtepu8_epi16(a));
						else if constexpr (std::is_same_v<From, int8_t> && sizeof(To) == 8) return TV(_mm512_cvtepi8_epi64(a));
						else if constexpr (std::is_same_v<From, int8_t> && sizeof(To) == 4) return TV(_mm512_cvtepi8_epi32(a));
						else if constexpr (std::is_same_v<From, int8_t> && sizeof(To) == 2) return TV(_mm512_cvtepi8_epi16(a));

						else if constexpr (std::is_same_v<From, uint16_t> && sizeof(To) == 8) return TV(_mm512_cvtepu16_epi64(a));
						else if constexpr (std::is_same_v<From, uint16_t> && sizeof(To) == 4) return TV(_mm512_cvtepu16_epi32(a));
						else if constexpr (std::is_same_v<From, int16_t> && sizeof(To) == 8) return TV(_mm512_cvtepi16_epi64(a));
						else if constexpr (std::is_same_v<From, int16_t> && sizeof(To) == 4) return TV(_mm512_cvtepi16_epi32(a));

						else if constexpr (std::is_same_v<From, uint32_t> && sizeof(To) == 8) return TV(_mm512_cvtepu32_epi64(a));
						else if constexpr (std::is_same_v<From, int32_t> && sizeof(To) == 8) return TV(_mm512_cvtepi32_epi64(a));
					}
					else static_assert(always_false_v<To, From>, "Unsupported arguments for SIMD_Vector zmm_cvt");
				}
				else if constexpr (ymm_sized<TV> || ymm_sized<FV>)
				{
					//Truncation is REQUIRED for FP -> int
					if constexpr (std::is_same_v<From, double> && std::is_same_v<To, uint64_t>)  return TV(_mm256_cvttpd_epu64(a));
					else if constexpr (std::is_same_v<From, double> && std::is_same_v<To, int64_t>)  return TV(_mm256_cvttpd_epi64(a));
					else if constexpr (std::is_same_v<From, double> && std::is_same_v<To, float>)  return TV(_mm256_cvtpd_ps(a));
					else if constexpr (std::is_same_v<From, double> && std::is_same_v<To, uint32_t>)  return TV(_mm256_cvttpd_epu32(a));
					else if constexpr (std::is_same_v<From, double> && std::is_same_v<To, int32_t>)  return TV(_mm256_cvttpd_epi32(a));

					else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, double>) return TV(_mm256_cvtps_pd(a));
					else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, uint64_t>) return TV(_mm256_cvttps_epu64(a));
					else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, int64_t>) return TV(_mm256_cvttps_epi64(a));
					else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, uint32_t>) return TV(_mm256_cvttps_epu32(a));
					else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, int32_t>) return TV(_mm256_cvttps_epi32(a));
					//else if constexpr (std::is_same_v<From, float16_t> && std::is_same_v<To, float>) return TV(_mm256_cvtph_ps(a));

					//only float conversions care about the signedness
					else if constexpr (std::is_same_v<From, uint64_t> && std::is_same_v<To, double>) return TV(_mm256_cvtepu64_pd(a));
					else if constexpr (std::is_same_v<From, uint64_t> && std::is_same_v<To, float>) return TV(_mm256_cvtepu64_ps(a));
					else if constexpr (std::is_same_v<From, int64_t> && std::is_same_v<To, double>) return TV(_mm256_cvtepi64_pd(a));
					else if constexpr (std::is_same_v<From, int64_t> && std::is_same_v<To, float>) return TV(_mm256_cvtepi64_ps(a));
					else if constexpr (std::is_same_v<From, uint32_t> && std::is_same_v<To, double>) return TV(_mm256_cvtepu32_pd(a));
					else if constexpr (std::is_same_v<From, uint32_t> && std::is_same_v<To, float>) return TV(_mm256_cvtepu32_ps(a));
					else if constexpr (std::is_same_v<From, int32_t> && std::is_same_v<To, double>) return TV(_mm256_cvtepi32_pd(a));
					else if constexpr (std::is_same_v<From, int32_t> && std::is_same_v<To, float>) return TV(_mm256_cvtepi32_ps(a));

					//integer conversion zone
					else if constexpr (std::is_integral_v<From> && std::is_integral_v<To>)
					{
						//conversion to smaller ints doesn't care about sign
						if constexpr (sizeof(From) == 8 && sizeof(To) == 4) return TV(_mm256_cvtepi64_epi32(a));
						else if constexpr (sizeof(From) == 8 && sizeof(To) == 2) return TV(_mm256_cvtepi64_epi16(a));
						else if constexpr (sizeof(From) == 8 && sizeof(To) == 1) return TV(_mm256_cvtepi64_epi8(a));
						else if constexpr (sizeof(From) == 4 && sizeof(To) == 2) return TV(_mm256_cvtepi32_epi16(a));
						else if constexpr (sizeof(From) == 4 && sizeof(To) == 1) return TV(_mm256_cvtepi32_epi8(a));
						else if constexpr (sizeof(From) == 2 && sizeof(To) == 1) return TV(_mm256_cvtepi16_epi8(a));

						//conversion to bigger DOES care about sign
						else if constexpr (std::is_same_v<From, uint8_t> && sizeof(To) == 8) return TV(_mm256_cvtepu8_epi64(a));
						else if constexpr (std::is_same_v<From, uint8_t> && sizeof(To) == 4) return TV(_mm256_cvtepu8_epi32(a));
						else if constexpr (std::is_same_v<From, uint8_t> && sizeof(To) == 2) return TV(_mm256_cvtepu8_epi16(a));
						else if constexpr (std::is_same_v<From, int8_t> && sizeof(To) == 8) return TV(_mm256_cvtepi8_epi64(a));
						else if constexpr (std::is_same_v<From, int8_t> && sizeof(To) == 4) return TV(_mm256_cvtepi8_epi32(a));
						else if constexpr (std::is_same_v<From, int8_t> && sizeof(To) == 2) return TV(_mm256_cvtepi8_epi16(a));

						else if constexpr (std::is_same_v<From, uint16_t> && sizeof(To) == 8) return TV(_mm256_cvtepu16_epi64(a));
						else if constexpr (std::is_same_v<From, uint16_t> && sizeof(To) == 4) return TV(_mm256_cvtepu16_epi32(a));
						else if constexpr (std::is_same_v<From, int16_t> && sizeof(To) == 8) return TV(_mm256_cvtepi16_epi64(a));
						else if constexpr (std::is_same_v<From, int16_t> && sizeof(To) == 4) return TV(_mm256_cvtepi16_epi32(a));

						else if constexpr (std::is_same_v<From, uint32_t> && sizeof(To) == 8) return TV(_mm256_cvtepu32_epi64(a));
						else if constexpr (std::is_same_v<From, int32_t> && sizeof(To) == 8) return TV(_mm256_cvtepi32_epi64(a));
					}
					else static_assert(always_false_v<To, From>, "Unsupported arguments for SIMD_Vector ymm_cvt");
				}
				else if constexpr (xmm_sized<TV> || xmm_sized<FV>)
				{
					//Truncation is REQUIRED for FP -> int
					if constexpr (std::is_same_v<From, double> && std::is_same_v<To, uint64_t>)  return TV(_mm_cvttpd_epu64(a));
					else if constexpr (std::is_same_v<From, double> && std::is_same_v<To, int64_t>)  return TV(_mm_cvttpd_epi64(a));
					else if constexpr (std::is_same_v<From, double> && std::is_same_v<To, float>)  return TV(_mm_cvtpd_ps(a));
					else if constexpr (std::is_same_v<From, double> && std::is_same_v<To, uint32_t>)  return TV(_mm_cvttpd_epu32(a));
					else if constexpr (std::is_same_v<From, double> && std::is_same_v<To, int32_t>)  return TV(_mm_cvttpd_epi32(a));

					else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, double>) return TV(_mm_cvtps_pd(a));
					else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, uint64_t>) return TV(_mm_cvttps_epu64(a));
					else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, int64_t>) return TV(_mm_cvttps_epi64(a));
					else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, uint32_t>) return TV(_mm_cvttps_epu32(a));
					else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, int32_t>) return TV(_mm_cvttps_epi32(a));
					//TODO: FP16 conversion
					//else if constexpr (std::is_same_v<From, float16_t> && std::is_same_v<To, float>) return TV(_mm_cvtph_ps(a));

					//only float conversions care about the signedness
					else if constexpr (std::is_same_v<From, uint64_t> && std::is_same_v<To, double>) return TV(_mm_cvtepu64_pd(a));
					else if constexpr (std::is_same_v<From, uint64_t> && std::is_same_v<To, float>) return TV(_mm_cvtepu64_ps(a));
					else if constexpr (std::is_same_v<From, int64_t> && std::is_same_v<To, double>) return TV(_mm_cvtepi64_pd(a));
					else if constexpr (std::is_same_v<From, int64_t> && std::is_same_v<To, float>) return TV(_mm_cvtepi64_ps(a));
					else if constexpr (std::is_same_v<From, uint32_t> && std::is_same_v<To, double>) return TV(_mm_cvtepu32_pd(a));
					else if constexpr (std::is_same_v<From, uint32_t> && std::is_same_v<To, float>) return TV(_mm_cvtepu32_ps(a));
					else if constexpr (std::is_same_v<From, int32_t> && std::is_same_v<To, double>) return TV(_mm_cvtepi32_pd(a));
					else if constexpr (std::is_same_v<From, int32_t> && std::is_same_v<To, float>) return TV(_mm_cvtepi32_ps(a));

					//integer conversion zone
					else if constexpr (std::is_integral_v<From> && std::is_integral_v<To>)
					{
						//conversion to smaller ints doesn't care about sign
						if constexpr (sizeof(From) == 8 && sizeof(To) == 4) return TV(_mm_cvtepi64_epi32(a));
						else if constexpr (sizeof(From) == 8 && sizeof(To) == 2) return TV(_mm_cvtepi64_epi16(a));
						else if constexpr (sizeof(From) == 8 && sizeof(To) == 1) return TV(_mm_cvtepi64_epi8(a));
						else if constexpr (sizeof(From) == 4 && sizeof(To) == 2) return TV(_mm_cvtepi32_epi16(a));
						else if constexpr (sizeof(From) == 4 && sizeof(To) == 1) return TV(_mm_cvtepi32_epi8(a));
						else if constexpr (sizeof(From) == 2 && sizeof(To) == 1) return TV(_mm_cvtepi16_epi8(a));

						//conversion to bigger DOES care about sign
						else if constexpr (std::is_same_v<From, uint8_t> && sizeof(To) == 8) return TV(_mm_cvtepu8_epi64(a));
						else if constexpr (std::is_same_v<From, uint8_t> && sizeof(To) == 4) return TV(_mm_cvtepu8_epi32(a));
						else if constexpr (std::is_same_v<From, uint8_t> && sizeof(To) == 2) return TV(_mm_cvtepu8_epi16(a));
						else if constexpr (std::is_same_v<From, int8_t> && sizeof(To) == 8) return TV(_mm_cvtepi8_epi64(a));
						else if constexpr (std::is_same_v<From, int8_t> && sizeof(To) == 4) return TV(_mm_cvtepi8_epi32(a));
						else if constexpr (std::is_same_v<From, int8_t> && sizeof(To) == 2) return TV(_mm_cvtepi8_epi16(a));

						else if constexpr (std::is_same_v<From, uint16_t> && sizeof(To) == 8) return TV(_mm_cvtepu16_epi64(a));
						else if constexpr (std::is_same_v<From, uint16_t> && sizeof(To) == 4) return TV(_mm_cvtepu16_epi32(a));
						else if constexpr (std::is_same_v<From, int16_t> && sizeof(To) == 8) return TV(_mm_cvtepi16_epi64(a));
						else if constexpr (std::is_same_v<From, int16_t> && sizeof(To) == 4) return TV(_mm_cvtepi16_epi32(a));

						else if constexpr (std::is_same_v<From, uint32_t> && sizeof(To) == 8) return TV(_mm_cvtepu32_epi64(a));
						else if constexpr (std::is_same_v<From, int32_t> && sizeof(To) == 8) return TV(_mm_cvtepi32_epi64(a));
					}
					else static_assert(always_false_v<To, From>, "Unsupported arguments for SIMD_Vector xmm_cvt");
				}
			}

			//syntactic sugar for other cvt. 
			template<typename T, size_t N, typename From>
			requires (T::IsSimdVector && T::LaneCount == N)
			static T cvt(const SIMD_Vector<From, N>& a)
			{
				return cvt<T::ScalarType>(a);
			}
		};
	}
}