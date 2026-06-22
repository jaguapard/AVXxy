#pragma once
#include "funcs.h"
#include "SIMD_Vector.h"
#include "Dispatcher.h"

namespace AVXXY_NAMESPACE
{
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> add(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::Dispatcher::run<internals::op_add>(a, b);
	}
	/*
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> sub(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_sub::eval(a, b);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> mul(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_mul::eval(a, b);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> div(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_div::eval(a, b);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> logic_and(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_and::eval(a, b);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> logic_or(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_or::eval(a, b);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> logic_xor(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_xor::eval(a, b);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> logic_not(const SIMD_Vector<S, N>& a)
	{
		return internals::op_not::eval(a);
	}
	template<typename S, size_t N, typename I>
	__forceinline SIMD_Vector<S, N> shift_left(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& amount)
	{
		return internals::op_shl::eval(a, amount);
	}
	template<typename S, size_t N, typename I>
	__forceinline SIMD_Vector<S, N> shift_right(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& amount)
	{
		return internals::op_shr::eval(a, amount);
	}
	template<typename S, size_t N, typename I>
	__forceinline SIMD_Vector<S, N> permx(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind)
	{
		return internals::op_permx::eval(a, ind);
	}
	template<typename S, size_t N, typename I>
	__forceinline SIMD_Vector<S, N> permx2(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const SIMD_Vector<I, N>& ind)
	{
		return internals::op_permx2::eval(a, b, ind);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<float, N> sqrtf(const SIMD_Vector<S, N>& a)
	{
		return internals::op_sqrtf::eval(a);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<double, N> sqrtd(const SIMD_Vector<S, N>& a)
	{
		return internals::op_sqrtd::eval(a);
	}*/


	template<typename To, size_t N, typename From>
	__forceinline SIMD_Vector<To, N> vcvt(const SIMD_Vector<From, N>& value)
	{
		return internals::Dispatcher::run<internals::op_cvt<To>>(value);
	}
	/*
	template<typename T, typename S, size_t N>
		requires (meta::IsSimdVector<T>)
	__forceinline T vcast(const SIMD_Vector<S, N>& value)
	{
		T ret;
		memcpy(&ret, &value, std::min(sizeof(ret), sizeof(value)));
		return ret;
	}
	template<typename T, typename S, size_t N>
	__forceinline T vreinterpret(const SIMD_Vector<S, N>& value)
	{
		T ret;
		memcpy(&ret, &value, std::min(sizeof(ret), sizeof(value)));
		return ret;
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> mask_mov(const SIMD_Vector<S, N>& ifBitClear, const mask_t<S, N>& mask, const SIMD_Vector<S, N>& ifBitSet)
	{
		return internals::op_mask_mov::eval(ifBitClear, mask, ifBitSet);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> maskz_mov(const mask_t<S, N>& mask, const SIMD_Vector<S, N>& ifBitSet)
	{
		return mask_mov(SIMD_Vector<S, N>(0), mask, ifBitSet);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> blend(const mask_t<S, N>& mask, const SIMD_Vector<S, N>& ifBitClear, const SIMD_Vector<S, N>& ifBitSet)
	{
		return mask_mov(ifBitClear, mask, ifBitSet);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> load(const void* p, const mask_t<S, N>& mask, const SIMD_Vector<S, N>& src)
	{
		return internals::op_load::eval<S, N>(p, mask, src);
	}
	template<typename S, size_t N>
	__forceinline void store(const SIMD_Vector<S, N>& v, void* p, const mask_t<S, N>& mask)
	{
		return internals::op_store::eval(v, p, mask);
	}

	template<typename S, size_t N, size_t Scale, typename I>
	__forceinline void scatter(const SIMD_Vector<S, N>& vec, void* base, const SIMD_Vector<I, N>& ind, const mask_t<S, N>& mask)
	{
		return internals::op_scatter::eval<S, N, Scale, I>(vec, base, ind, mask);
	}
	template<typename S, size_t N>
	__forceinline mask_t<S, N> cmp_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_cmpeq::eval(a, b);
	}
	template<typename S, size_t N>
	__forceinline mask_t<S, N> cmp_not_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_cmpneq::eval(a, b);
	}
	template<typename S, size_t N>
	__forceinline mask_t<S, N> cmp_less(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_cmplt::eval(a, b);
	}
	template<typename S, size_t N>
	__forceinline mask_t<S, N> cmp_less_or_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_cmple::eval(a, b);
	}
	template<typename S, size_t N>
	__forceinline mask_t<S, N> cmp_greater(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_cmpgt::eval(a, b);
	}
	template<typename S, size_t N>
	__forceinline mask_t<S, N> cmp_greater_or_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_cmpge::eval(a, b);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> abs(const SIMD_Vector<S, N>& a)
	{
		return internals::op_abs::eval(a);
	}

	template<typename S, size_t N>
		requires (meta::any_float<S>)
	__forceinline SIMD_Vector<S, N> floor(const SIMD_Vector<S, N>& a)
	{
		return internals::op_floor::eval(a);
	}
	template<typename S, size_t N>
		requires (meta::any_float<S>)
	__forceinline SIMD_Vector<S, N> ceil(const SIMD_Vector<S, N>& a)
	{
		return internals::op_ceil::eval(a);
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> min(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_min::eval(a, b);
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> max(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_max::eval(a, b);
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> clamp(const SIMD_Vector<S, N>& val, const SIMD_Vector<S, N>& min, const SIMD_Vector<S, N>& max)
	{
		return AVXXY_NAMESPACE::max(min, AVXXY_NAMESPACE::min(val, max));
	}


	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> unpacklo(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_unpacklo::eval(a, b);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> unpackhi(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::op_unpackhi::eval(a, b);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> compress(const mask_t<S, N>& mask, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& src)
	{
		return internals::op_compress::eval(mask, a, src);
	}
	template<typename S, size_t N>
		requires (sizeof(S) * 8 >= N)
	__forceinline SIMD_Vector<typename meta::ScalarTraits<S>::UintT, N> conflict(const SIMD_Vector<S, N>& a)
	{
		return internals::op_conflict::eval(a);
	}

	template<typename S, size_t N>
	mask_t<S, N> movemask(const SIMD_Vector<S, N>& v)
	{
		return internals::op_movemask::eval(v);
	}

	template<typename S, size_t N, meta::ScalarSizeClassEnum C>
	SIMD_Vector<S, N> movm(const SIMD_Mask<C, N>& mask)
	{
		return internals::op_movm::eval<S, N>(mask);
	}

	template<typename S, size_t N, size_t Scale, typename I>
	__forceinline SIMD_Vector<S, N> __gather_impl(const void* base, const SIMD_Vector<I, N>& ind, const typename SIMD_Vector<S, N>::MaskT& mask, const SIMD_Vector<S, N>& src)
	{
		return internals::op_gather::eval<S, N, Scale>(base, ind, mask, src);
	}*/
}