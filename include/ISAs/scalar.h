#pragma once
#include "../namespace.h"
#include "../tags.h"
#include "../SIMD_BitMask.h"
#include "../SIMD_Vector.h"
namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		namespace ISA
		{
			struct Scalar
			{
				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_add, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					SIMD_Vector<S, N> ret;
					for (size_t i = 0; i < N; ++i) ret[i] = a[i] + b[i];
					return ret;
				}
			};
		}
	}
}