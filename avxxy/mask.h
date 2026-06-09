#pragma once
#include "namespace.h"
#include "backends.h"
namespace AVXXY_NAMESPACE
{
	template<size_t N>
	struct SIMD_Mask
	{
	public:
		static inline constexpr size_t BitCount = N;
	private:
		//bit mask
	};
}