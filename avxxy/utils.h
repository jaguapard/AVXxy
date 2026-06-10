#pragma once
#include "namespace.h"
#include <cstddef>
namespace AVXXY_NAMESPACE
{
	namespace utils
	{
		static constexpr bool isPowerOf2(std::size_t N)
		{
			if (N <= 2) return true;
			if (N % 2 != 0) return false;
			return isPowerOf2(N / 2);
		}
		static constexpr bool inRange(std::size_t val, std::size_t min, std::size_t max)
		{
			return val >= min && val <= max;
		}
	}
}