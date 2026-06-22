#pragma once
#include "shared.h"
namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		struct op_conflict : OperationBase
		{
			template<typename S, size_t N>
				requires (sizeof(S) * 8 >= N)
			static SIMD_Vector<typename ScalarTraits<S>::UintT, N> run(const SIMD_Vector<S, N>& a)
			{
				using U = ScalarTraits<S>::UintT;
				using T = SIMD_Vector<U, N>;
				T ret;
				for (size_t i = 0; i < N; ++i)
				{
					U acc = 0;
					for (size_t j = 0; j < i; ++j)
					{
						if (a[i] == a[j]) acc |= U(1) << j;
					}
					ret[i] = acc;
				}
				return ret;
			}
		};
	}
}