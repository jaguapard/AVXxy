#pragma once
#include "funcs.h"
#include "Dispatcher.h"

namespace AVXXY_NAMESPACE
{
	template<typename S, size_t N>
	inline SIMD_Vector<S, N> AVXXY_NAMESPACE::add(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::DefaultDispatcher::run(internals::op_add{}, a, b);
	}
}