#pragma once
#include "../vector.h"
#include "avx2.h"

namespace AVXXY_NAMESPACE
{
	namespace backends
	{
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