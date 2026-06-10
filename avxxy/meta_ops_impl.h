#pragma once
#include "meta_ops.h"

namespace AVXXY_NAMESPACE
{
	template<typename To, typename S, size_t N>
	requires (To::IsSimdVector)
	inline To AVXXY_NAMESPACE::vcast(const SIMD_Vector<S, N>& a)
	{
		To ret;
		memcpy(&ret, &a, std::min(sizeof(ret), sizeof(a)));
		return ret;
	}
}