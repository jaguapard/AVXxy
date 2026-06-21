#pragma once
#include "namespace.h"
#include "meta/meta.h"
#include "FeatureSet.h"

namespace AVXXY_NAMESPACE
{
	template<typename S, size_t N>
		requires meta::IsValid_SIMD_Vector<S, N>
	class SIMD_Vector;

	template<size_t N>
	concept IsValid_SIMD_Mask = N >= 2 && N <= 64 && meta::isPowerOf2(N);

	template<meta::ScalarSizeClassEnum LS, size_t N>
	requires IsValid_SIMD_Mask<N>
	class SIMD_Mask;


	template<meta::ScalarSizeClassEnum LS, size_t N>
		requires IsValid_SIMD_Mask<N>
	class SIMD_Mask
	{
		template <concepts::LaneSizeEnum FriendS, size_t FriendN>
			requires IsValid_SIMD_Mask<N>
		friend class SIMD_Mask;


	};
}