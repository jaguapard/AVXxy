#pragma once

#include <algorithm>
#include <array>
#include <iostream>
#include <cstring>
#include "meta/meta.h"

namespace AVXXY_NAMESPACE
{
	template <typename S, size_t N>
		requires meta::IsValid_SIMD_Vector<S, N>
	class alignas(std::min<uint32_t>(64, sizeof(S)* N)) SIMD_Vector
	{
	private:
		std::array<S, N> arr;

	public:
		SIMD_Vector() {};
		const S& operator[](size_t i) const { return arr[i]; }
		S& operator[](size_t i) { return arr[i]; }

		//Returns vector filled with sequential values (value == lane index, like 0, 1, 2, ..., N-1)
		static SIMD_Vector<S, N> iota()
		{
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = i;
			return ret;
		}
		//Copies and returns lower half of this vector
		SIMD_Vector<S, N / 2> lo() const
			requires (N >= 4)
		{
			SIMD_Vector<S, N / 2> ret;
			memcpy(ret.arr.data(), arr.data(), sizeof(ret));
			return ret;
		}
		//Copies and returns upper half of this vector
		SIMD_Vector<S, N / 2> hi() const
			requires (N >= 4)
		{
			SIMD_Vector<S, N / 2> ret;
			memcpy(ret.arr.data(), arr.data() + N / 2, sizeof(ret));
			return ret;
		}
		//Copies and returns lower half of this vector
		S lo() const
			requires (N == 2)
		{
			return arr[0];
		}
		//Copies and returns upper half of this vector
		S hi() const
			requires (N == 2)
		{
			return arr[1];
		}

	};

	template<typename S, size_t N>
	std::ostream& operator<<(std::ostream& os, const SIMD_Vector<S, N>& a)
	{
		for (size_t i = 0; i < N; ++i)
		{
			if constexpr (meta::any_i8<S>) os << int(a[i]);
			else os << a[i];
			if (i < N - 1) os << " ";
		}
		return os;
	}
}