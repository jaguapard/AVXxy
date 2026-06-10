#pragma once
#include "namespace.h"
#include "concepts.h"

namespace AVXXY_NAMESPACE
{
	template<size_t N>
	struct SIMD_Mask
	{
	public:
		static_assert(N >= 2);
		static_assert(N < 64);
		static_assert(utils::isPowerOf2(N));

		static inline constexpr size_t BitCount = N;
		using UintT = typename concepts::bits_to_uint_t<N>::type;
		static inline constexpr UintT AllOnes = (N == sizeof(UintT) * 8) ? ~UintT(0) : ((UintT(1) << N) - 1);

		SIMD_Mask(UintT value);
		operator UintT() const;

		//Returns true if bit i is set, false otherwise. Cannot be used to modify mask bits, for that use setBit
		bool operator[](size_t i) const;
		
		//Sets the bit i of the mask to 1 if value is true, or 0 otherwise
		void setBit(size_t i, bool value);

		SIMD_Mask<N / 2> lo() const;
		SIMD_Mask<N / 2> hi() const;

		SIMD_Mask<N>& operator&=(const SIMD_Mask<N>& other);
		SIMD_Mask<N>& operator|=(const SIMD_Mask<N>& other);
		SIMD_Mask<N>& operator^=(const SIMD_Mask<N>& other);
	private:
		UintT underlying;
	};
}