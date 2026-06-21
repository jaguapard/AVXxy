#pragma once
#include "funcs.h"
#include "operations/all.h"

namespace AVXXY_NAMESPACE
{
	template<typename S, size_t N>
	SIMD_Vector<S, N> add(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_add::run(a, b);
	}
	template<typename S, size_t N>
	SIMD_Vector<S, N> sub(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_sub::run(a, b);
	}
}