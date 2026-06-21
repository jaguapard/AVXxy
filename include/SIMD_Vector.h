#pragma once

#include <algorithm>
#include <array>
#include <iostream>
#include <cstring>

namespace AVXXY_NAMESPACE
{
	template <typename S, size_t N>
	class alignas(std::min<uint32_t>(64, sizeof(S)* N)) SIMD_Vector
	{
	public:

	private:
		std::array<S, N> arr;
	};

	template<typename S, size_t N>
	std::ostream& operator<<(std::ostream& os, const SIMD_Vector<S, N>& a)
	{
		for (size_t i = 0; i < N; ++i)
		{
			if constexpr (concepts::any_i8<S>) os << int(a[i]);
			else os << a[i];
			if (i < N - 1) os << " ";
		}
		return os;
	}
}