#pragma once
#include "mask.h"
#include "concepts.h"
#include <algorithm>

namespace AVXXY_NAMESPACE
{
	template<typename _S, size_t _N> concept IsValid_SIMD_Vector = _N >= 2 && _N <= 64 && utils::isPowerOf2(_N) && concepts::IsScalarType<_S>; //for now, bigger than 64 lanes vectors are not supported (mainly due to mask type not being ready for it)

	template<typename _S, size_t _N, typename Backend = backends::current>
		requires IsValid_SIMD_Vector<_S, _N>
	struct alignas(std::min<uint32_t>(64, sizeof(_S)* _N)) SIMD_Vector
	{
		using ScalarType = _S;

		static inline constexpr size_t LaneCount = _N;
		static inline constexpr size_t ByteSize = sizeof(ScalarType) * LaneCount;
		static inline constexpr bool IsSimdVector = true;
	};
}