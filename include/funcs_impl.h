#pragma once
#include "funcs.h"
#include "Dispatcher.h"

namespace AVXXY_NAMESPACE
{
	template<typename S, size_t N>
	inline SIMD_Vector<S, N> add(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::DefaultDispatcher::run(internals::op_add{}, a, b);
	}
	template<typename S, size_t N>
	inline SIMD_Vector<S, N> sub(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::DefaultDispatcher::run(internals::op_sub{}, a, b);
	}
	template<typename S, size_t N>
	inline SIMD_Vector<S, N> mul(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::DefaultDispatcher::run(internals::op_mul{}, a, b);
	}
	template<typename S, size_t N>
	inline SIMD_Vector<S, N> div(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::DefaultDispatcher::run(internals::op_div{}, a, b);
	}
	template<typename S, size_t N>
	inline SIMD_Vector<S, N> logic_and(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::DefaultDispatcher::run(internals::op_and{}, a, b);
	}
	template<typename S, size_t N>
	inline SIMD_Vector<S, N> logic_or(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::DefaultDispatcher::run(internals::op_or{}, a, b);
	}
	template<typename S, size_t N>
	inline SIMD_Vector<S, N> logic_xor(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::DefaultDispatcher::run(internals::op_xor{}, a, b);
	}
	template<typename S, size_t N>
	inline SIMD_Vector<S, N> logic_not(const SIMD_Vector<S, N>& a)
	{
		return internals::DefaultDispatcher::run(internals::op_not{}, a);
	}
	
}