#pragma once
#include "../SIMD_Vector.h"
#include "../Dispatcher.h"
#include "../op_tags.h"
namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		struct ISA_Scalar
		{
			static inline constexpr FeatureSet FS = internals::FS_current;
			template<op_add, typename S, size_t N>
			static SIMD_Vector<S, N> eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i] + b[i];
				return ret;
			}
		};
	}
}