#pragma once
#include "mask.h"
#include "concepts.h"
#include <algorithm>
#include <array>

namespace AVXXY_NAMESPACE
{
	template<typename _S, size_t _N> concept IsValid_SIMD_Vector = _N >= 2 && _N <= 64 && utils::isPowerOf2(_N) && concepts::IsScalarType<_S>; //for now, bigger than 64 lanes vectors are not supported (mainly due to mask type not being ready for it)

	template<typename Vec, typename IntrinVec>
	concept ConversionToNativeVectorLegal =
		(concepts::xmm_sized<Vec> && std::is_same_v<IntrinVec, typename concepts::reg128<typename Vec::ScalarType>::type>) ||
		(concepts::ymm_sized<Vec> && std::is_same_v<IntrinVec, typename concepts::reg256<typename Vec::ScalarType>::type>) ||
		(concepts::zmm_sized<Vec> && std::is_same_v<IntrinVec, typename concepts::reg512<typename Vec::ScalarType>::type>);

	template<typename _S, size_t _N, typename Backend = backends::current>
		requires IsValid_SIMD_Vector<_S, _N>
	struct alignas(std::min<uint32_t>(64, sizeof(_S)* _N)) SIMD_Vector
	{
		using ScalarType = _S;
		using Self = SIMD_Vector<_S, _N>;

		static inline constexpr size_t LaneCount = _N;
		static inline constexpr size_t ByteSize = sizeof(ScalarType) * LaneCount;
		static inline constexpr bool IsSimdVector = true;

		const ScalarType& operator[](size_t i) const { return arr[i]; }
		ScalarType& operator[](size_t i) { return arr[i]; }
		
		SIMD_Vector<ScalarType, LaneCount / 2> lo() const 
		{
			SIMD_Vector<ScalarType, LaneCount / 2> ret;
			memcpy(ret.arr.data(), arr.data(), sizeof(ret));
			return ret;
		}
		SIMD_Vector<ScalarType, LaneCount / 2> hi() const
		{
			SIMD_Vector<ScalarType, LaneCount / 2> ret;
			memcpy(ret.arr.data(), arr.data() + LaneCount / 2, sizeof(ret));
			return ret;
		}

		//Broadcasts a scalar value to all lanes of vector. The input value is converted to vector's intrinsic type before broadcasting
		template<typename T> requires concepts::IsScalarType<T>
		__forceinline SIMD_Vector(const T& s) { for (size_t i = 0; i < LaneCount; ++i) (*this)[i] = s; }

		//Constructs vector from it's intrinsic type. The intrinsic vector type must be of the same size class as constructed vector:
		//Vectors less than 17 bytes can be constructed from 128 bit intrinsic types.
		//Vectors between 17 and 32 bytes can be constructed from 256 bit intrinsic types.
		//Vectors between 33 and 64 bytes can be constructed from 512 bit intrinsic types.
		//Integral intrinsic vectors can be used to construct any integral SIMD_Vector of same size class
		//Floating point vectors require the scalar type of intrinsic vector and SIMD_Vector to match
		//If SIMD_Vector and intrinsic vector sizes mismatch, only the lowest bits of intrinsic vector are copied to the constructed SIMD_Vector
		template <typename T>
		__forceinline SIMD_Vector(const T& intrinsicVec) requires(concepts::IsIntrinsicVector<T>&& ConversionToNativeVectorLegal<Self, T>)
		{
			memcpy(arr.data(), &intrinsicVec, std::min(sizeof(Self), sizeof(T)));
		}

		//Represents vector as it's intrinsic type.  The intrinsic vector type is in the same size class as this vector:
		//Vectors less than 17 bytes: 128 bit intrinsic types.
		//Vectors between 17 and 32 bytes: 256 bit intrinsic types.
		//Vectors between 33 and 64 bytes: 512 bit intrinsic types.
		//Integral SIMD_Vectors convert to integral intrinsic vectors
		//Floating point SIMD_Vectors convert to intrinsic vectors of same scalar type
		//If SIMD_Vector and intrinsic vector sizes mismatch, upper bits of the returned vector are undefined
		template <typename T>
		__forceinline operator T() const requires(concepts::IsIntrinsicVector<T>&& ConversionToNativeVectorLegal<Self, T>)
		{
			T ret;
			memcpy(&ret, arr.data(), std::min(sizeof(T), sizeof(Self)));
			return ret;
		}
	private:
		std::array<ScalarType, LaneCount> arr;
		//ScalarType arr[LaneCount];
	};
}