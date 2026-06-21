#pragma once
#include "../namespace.h"
#include "enums.h"
#include <stdint.h>
#include <immintrin.h>
#include "../small_fp.h"

namespace AVXXY_NAMESPACE
{
	namespace meta
	{
		//template <typename T>
		//concept SupportsSizeClass = 
		template <typename T, typename... Ts> inline constexpr bool is_any_of_v = (std::is_same_v<T, Ts> || ...);

		template<typename T> concept IsScalarType = is_any_of_v<T, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double, fp16_t, bf16_t>;
		template<typename... Ts> concept AllAreScalarTypes = (IsScalarType<Ts> && ...);
		template<typename T> concept IsIntrinsicVector = is_any_of_v<T, __m128i, __m128, __m128d, __m256i, __m256, __m256d, __m512i, __m512, __m512d>;
	}
}