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

	template<typename S, size_t N, typename I>
	inline SIMD_Vector<S, N> shift_left(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& amount)
	{
		return internals::DefaultDispatcher::run(internals::op_shl{}, a, amount);
	}
	template<typename S, size_t N, typename I>
	inline SIMD_Vector<S, N> shift_right(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& amount)
	{
		return internals::DefaultDispatcher::run(internals::op_shr{}, a, amount);
	}

	template<typename S, size_t N, typename I>
	inline SIMD_Vector<S, N> permx(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind)
	{
		return internals::DefaultDispatcher::run(internals::op_permx{}, a, ind);
	}

	template<typename S, size_t N, typename I>
	inline SIMD_Vector<S, N> permx2(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const SIMD_Vector<I, N>& ind)
	{
		return internals::DefaultDispatcher::run(internals::op_permx2{}, a, b, ind);
	}

	template<typename S, size_t N>
	inline SIMD_Vector<float, N> sqrtf(const SIMD_Vector<S, N>& a)
	{
		return internals::DefaultDispatcher::run(internals::op_sqrtf{}, a);
	}

	template<typename S, size_t N>
	inline SIMD_Vector<double, N> sqrtd(const SIMD_Vector<S, N>& a)
	{
		return internals::DefaultDispatcher::run(internals::op_sqrtd{}, a);
	}

	template<typename To, size_t N, typename From>
	inline SIMD_Vector<To, N> vcvt(const SIMD_Vector<From, N>& value)
	{
		return internals::DefaultDispatcher::run(internals::op_cvt<To>{}, value);
	}

	template<typename S, size_t N>
	inline SIMD_Vector<S, N> abs(const SIMD_Vector<S, N>& a)
	{
		return internals::DefaultDispatcher::run(internals::op_abs{}, a);
	}
	
	template<typename T, typename S, size_t N>
	//	requires (T::IsSimdVector)
	inline T vcast(const SIMD_Vector<S, N>& value)
	{
		T ret;
		memcpy(&ret, &value, std::min(sizeof(ret), sizeof(value)));
		return ret;
	}

}