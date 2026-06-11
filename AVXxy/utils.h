#pragma once
#include "namespace.h"
namespace AVXXY_NAMESPACE
{
	namespace utils
	{
		static constexpr bool isPowerOf2(size_t N)
		{
			if (N <= 2) return true;
			if (N % 2 != 0) return false;
			return isPowerOf2(N / 2);
		}
		static constexpr bool inRange(size_t val, size_t min, size_t max)
		{
			return val >= min && val <= max;
		}
	}
}