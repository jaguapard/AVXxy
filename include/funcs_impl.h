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
	template<typename S, size_t N>
	SIMD_Vector<float, N> sqrtf(const SIMD_Vector<S, N>& a)
	{
		return internals::op_sqrtf::run(a);
	}
	template<typename S, size_t N>
	SIMD_Vector<double, N> sqrtd(const SIMD_Vector<S, N>& a)
	{
		return internals::op_sqrtd::run(a);
	}
	template<typename To, size_t N, typename From>
	SIMD_Vector<To, N> vcvt(const SIMD_Vector<From, N>& value)
	{
		return internals::op_cvt::run<To>(value);
	}
	template<typename T, typename S, size_t N>
	T vcast(const SIMD_Vector<S, N>& value)
	{
		T ret;
		memcpy(&ret, &value, std::min(sizeof(ret), sizeof(value)));
		return ret;
	}
	template<typename T, typename S, size_t N>
	T vreinterpret(const SIMD_Vector<S, N>& value)
	{
		T ret;
		memcpy(&ret, &value, std::min(sizeof(ret), sizeof(value)));
		return ret;
	}
	template<typename S, size_t N>
	SIMD_Vector<S, N> mask_mov(const SIMD_Vector<S, N>& ifBitClear, const mask_t<S, N>& mask, const SIMD_Vector<S, N>& ifBitSet)
	{
		return internals::op_mask_mov::run(ifBitClear, mask, ifBitSet);
	}
	template<typename S, size_t N>
	SIMD_Vector<S, N> maskz_mov(const mask_t<S, N>& mask, const SIMD_Vector<S, N>& ifBitSet)
	{
		return mask_mov(SIMD_Vector<S, N>(0), mask, ifBitSet);
	}
	template<typename S, size_t N>
	SIMD_Vector<S, N> blend(const mask_t<S, N>& mask, const SIMD_Vector<S, N>& ifBitClear, const SIMD_Vector<S, N>& ifBitSet)
	{
		return mask_mov(ifBitClear, mask, ifBitSet);
	}
	template<typename S, size_t N>
	SIMD_Vector<S, N> load(const void* p, const mask_t<S, N>& mask, const SIMD_Vector<S, N>& src)
	{
		return internals::op_load::run<S, N>(p, mask, src);
	}
	template<typename S, size_t N>
	void store(const SIMD_Vector<S, N>& v, void* p, const mask_t<S, N>& mask)
	{
		return internals::op_store::run(v, p, mask);
	}

	template<typename S, size_t N, size_t Scale, typename I>
	void scatter(const SIMD_Vector<S, N>& vec, void* base, const SIMD_Vector<I, N>& ind, const mask_t<S, N>& mask)
	{
		return internals::op_scatter::run(vec, base, ind, mask);
	}
	template<typename S, size_t N>
	mask_t<S, N> cmp_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_cmpeq::run(a, b);
	}
	template<typename S, size_t N>
	mask_t<S, N> cmp_not_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_cmpneq::run(a, b);
	}
	template<typename S, size_t N>
	mask_t<S, N> cmp_less(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_cmplt::run(a, b);
	}
	template<typename S, size_t N>
	mask_t<S, N> cmp_less_or_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_cmple::run(a, b);
	}
	template<typename S, size_t N>
	mask_t<S, N> cmp_greater(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_cmpgt::run(a, b);
	}
	template<typename S, size_t N>
	mask_t<S, N> cmp_greater_or_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_cmpge::run(a, b);
	}
}