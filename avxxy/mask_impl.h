#pragma once
#include "mask.h"
#include <iostream>

namespace AVXXY_NAMESPACE
{
	//AllOnes anding is to clear out garbage from upper unused bits
	template<size_t N>
	__forceinline AVXXY_NAMESPACE::SIMD_Mask<N>::SIMD_Mask(UintT value)
	{
		underlying = value & AllOnes;
	}
	template<size_t N>
	inline SIMD_Mask<N>::operator UintT() const
	{
		return underlying & AllOnes;
	}
	template<size_t N>
	inline bool SIMD_Mask<N>::operator[](size_t i) const
	{
		return underlying & (UintT(1) << i);
	}

	template<size_t N>
	inline void SIMD_Mask<N>::setBit(size_t i, bool value)
	{
		underlying &= ~(UintT(1) << i); //clear bit i
		underlying |= UintT(value) << i;
		underlying &= AllOnes;
	}
	template<size_t N>
	inline SIMD_Mask<N / 2> SIMD_Mask<N>::lo() const
	{
		return underlying;
	}
	template<size_t N>
	inline SIMD_Mask<N / 2> SIMD_Mask<N>::hi() const
	{
		return underlying >> (N / 2);
	}
	template<size_t N>
	inline SIMD_Mask<N>& SIMD_Mask<N>::operator&=(const SIMD_Mask<N>& other)
	{
		*this = *this & other;
		return *this;
	}
	template<size_t N>
	inline SIMD_Mask<N>& SIMD_Mask<N>::operator|=(const SIMD_Mask<N>& other)
	{
		*this = *this | other;
		return *this;
	}
	template<size_t N>
	inline SIMD_Mask<N>& SIMD_Mask<N>::operator^=(const SIMD_Mask<N>& other)
	{
		*this = *this ^ other;
		return *this;
	}

	template <size_t N>
	static std::ostream& operator<<(std::ostream& os, const SIMD_Mask<N>& mask)
	{
		for (int i = 0; i < N; ++i) os << (mask[i] ? 1 : 0) << ",";
		os << (mask[N - 1] ? 1 : 0);
		return os;
	}
}