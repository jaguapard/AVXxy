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
	public:
		template <meta::ScalarSizeClassEnum FriendLS, size_t FriendN>
			requires IsValid_SIMD_Mask<N>
		friend class SIMD_Mask;

		using UintT = meta::ScalarSizeTraits<LS>::UintT;
		using IntT = meta::ScalarSizeTraits<LS>::IntT;
		using VecT = SIMD_Vector<IntT, N>;

		static inline constexpr bool IsBitMask = internals::FS_current.has(internals::AVX512_F) || !internals::FS_current.has(internals::SSE);
		static inline constexpr bool IsVectorMask = !IsBitMask;
		static inline constexpr UintT AllOnesUint = (N == sizeof(UintT) * 8) ? ~UintT(0) : ((UintT(1) << N) - 1);
		
		SIMD_Mask() {};
		SIMD_Mask(UintT bits);
		SIMD_Mask(const SIMD_Mask<LS, N / 2>& lo, const SIMD_Mask<LS, N / 2>& hi);

		template <meta::ScalarSizeClassEnum LS2>
		SIMD_Mask(const SIMD_Mask<LS2, N>& other);

		//Returns true if bit i is set, false otherwise. Cannot be used to modify mask bits, for that use setBit
		bool operator[](size_t i) const;

		//Sets the bit i of the mask to 1 if value is true, or 0 otherwise
		void setBit(size_t i, bool value);

		SIMD_Mask<LS, N / 2> lo() const;
		SIMD_Mask<LS, N / 2> hi() const;

		SIMD_Mask<LS, N> operator&(const SIMD_Mask<LS, N>& other) const;
		SIMD_Mask<LS, N> operator|(const SIMD_Mask<LS, N>& other) const;
		SIMD_Mask<LS, N> operator^(const SIMD_Mask<LS, N>& other) const;
		SIMD_Mask<LS, N> operator~() const;
		SIMD_Mask<LS, N>& operator&=(const SIMD_Mask<LS, N>& other);
		SIMD_Mask<LS, N>& operator|=(const SIMD_Mask<LS, N>& other);
		SIMD_Mask<LS, N>& operator^=(const SIMD_Mask<LS, N>& other);
	private:
		std::conditional_t<IsBitMask, UintT, VecT> underlying;
	};

	template<typename S, size_t N>
	using mask_t = SIMD_Mask<meta::ScalarTraits<S>::size_class, N>;
}