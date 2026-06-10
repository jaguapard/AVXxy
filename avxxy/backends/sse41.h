#pragma once
#include "../vector.h"
#include "sse2.h"

namespace AVXXY_NAMESPACE
{
	namespace backends
	{
		struct SSE41
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> add(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				return SSE2::add(a, b);
			}
		};
	}
}