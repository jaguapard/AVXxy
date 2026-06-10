#pragma once
#include "../vector.h"

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

			template<typename To, size_t N, typename From>
			static SIMD_Vector<To, N> cvt(const SIMD_Vector<From, N>& a)
			{
				SIMD_Vector<To, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i];
				return ret;
			}
		};
	}
}