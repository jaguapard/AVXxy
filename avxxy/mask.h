#pragma once
#include "namespace.h"
#include "backends.h"
namespace AVXXY_NAMESPACE
{
	template<typename _S, size_t _N, typename Backend = backends::current>
	struct SIMD_Mask
	{
	public:
		static inline constexpr size_t BitCount = _N;
		using ScalarType = _S;
	private:
	};
}