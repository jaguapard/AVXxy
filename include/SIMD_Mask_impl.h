#pragma once
#include "SIMD_Mask.h"
#include <iostream>
#include "funcs.h"

namespace AVXXY_NAMESPACE
{
	template<typename S, size_t N>
	inline constexpr SIMD_Mask<S, N> SIMD_Mask<S, N>::AllOnes()
	{
		return (N == sizeof(UintT) * 8) ? ~UintT(0) : ((UintT(1) << N) - 1);
	}
	template<typename S, size_t N>
	inline SIMD_Mask<S, N>::SIMD_Mask(UintT bits)
	{
		if constexpr (IsBitMask) this->underlying = bits & AllOnesUint;
		else
		{
			//TODO: bits to vector mask conversion here!
			//this->underlying = Default
		}
	}

	template<typename S, size_t N>
	inline bool SIMD_Mask<S, N>::operator[](size_t i) const
	{
		using U = typename VecT::UintScalarT;
		if constexpr (IsBitMask) return underlying & (UintT(1) << i) & AllOnesUint;
		else return vec2mask(underlying) & (UintT(1) << i) & AllOnesUint; //std::bit_cast<U>(underlying[i]) & (U(1) << (sizeof(U) * 8 - 1)); //TODO: clean?
	}
}