#pragma once
#include "vector.h"

namespace AVXXY_NAMESPACE
{
	// Reinterprets argument vector as another SIMD_Vector and returns the result
	// If returned vector's size is smaller than argument vector, only the lower bits of arguments are copied into the returned vector. 
	// If returned vector is larger than argument vector, then upper bits of the returned vector are undefined.
	template<typename To, typename S, size_t N> 
		requires (To::IsSimdVector)
	To vcast(const SIMD_Vector<S, N>& a);
}