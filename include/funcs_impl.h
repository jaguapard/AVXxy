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
	template<typename S, size_t N>
	SIMD_Vector<S, N> mul(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_mul::run(a, b);
	}
	template<typename S, size_t N>
	SIMD_Vector<S, N> div(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_div::run(a, b);
	}
	template<typename S, size_t N>
	SIMD_Vector<S, N> logic_and(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_and::run(a, b);
	}
	template<typename S, size_t N>
	SIMD_Vector<S, N> logic_or(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_or::run(a, b);
	}
	template<typename S, size_t N>
	SIMD_Vector<S, N> logic_xor(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_xor::run(a, b);
	}
	template<typename S, size_t N>
	SIMD_Vector<S, N> logic_not(const SIMD_Vector<S, N>& a)
	{
		return internals::op_not::run(a);
	}
	template<typename S, size_t N, typename I>
	SIMD_Vector<S, N> shift_left(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& amount)
	{
		return internals::op_shl::run(a, amount);
	}
	template<typename S, size_t N, typename I>
	SIMD_Vector<S, N> shift_right(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& amount)
	{
		return internals::op_shr::run(a, amount);
	}
	template<typename S, size_t N, typename I>
	SIMD_Vector<S, N> permx(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind)
	{
		return internals::op_permx::run(a, ind);
	}
	template<typename S, size_t N, typename I>
	SIMD_Vector<S, N> permx2(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const SIMD_Vector<I, N>& ind)
	{
		return internals::op_permx2::run(a, b, ind);
	}
}