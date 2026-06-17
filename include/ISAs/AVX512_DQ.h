#pragma once
#include "../namespace.h"
#include "../tags.h"
#include "../SIMD_BitMask.h"
#include "../SIMD_Vector.h"
#include "../FeatureSet.h"
namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		namespace ISA
		{
			using namespace concepts;
			using namespace utils;
			template<FeatureSet FS>
			struct AVX512DQ
			{
				template<typename S, size_t N>
				requires (sizeof(SIMD_Vector<S,N>) > 32 && (is_f32<S> || is_f64<S>))
				static SIMD_Vector<S, N> eval(op_and, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					if constexpr (sizeof(SIMD_Vector<S, N>) > 64) return { logic_and(a.lo(),b.lo()), logic_and(a.hi(),b.hi()) };
					else if constexpr (zmm_sized<SIMD_Vector<S, N>> && is_f64<S>) return _mm512_and_pd(a, b);
					else if constexpr (zmm_sized<SIMD_Vector<S, N>> && is_f32<S>) return _mm512_and_ps(a, b);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 32 && (is_f32<S> || is_f64<S>))
				static SIMD_Vector<S, N> eval(op_or, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					if constexpr (sizeof(SIMD_Vector<S, N>) > 64) return { logic_or(a.lo(),b.lo()), logic_or(a.hi(),b.hi()) };
					else if constexpr (zmm_sized<SIMD_Vector<S, N>> && is_f64<S>) return _mm512_or_pd(a, b);
					else if constexpr (zmm_sized<SIMD_Vector<S, N>> && is_f32<S>) return _mm512_or_ps(a, b);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 32 && (is_f32<S> || is_f64<S>))
				static SIMD_Vector<S, N> eval(op_xor, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					if constexpr (sizeof(SIMD_Vector<S, N>) > 64) return { logic_xor(a.lo(),b.lo()), logic_xor(a.hi(),b.hi()) };
					else if constexpr (zmm_sized<SIMD_Vector<S, N>> && is_f64<S>) return _mm512_xor_pd(a, b);
					else if constexpr (zmm_sized<SIMD_Vector<S, N>> && is_f32<S>) return _mm512_xor_ps(a, b);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 32)
				static SIMD_Vector<S, N> eval(op_not, const SIMD_Vector<S, N>& a)
				{
					using U = same_size_uint_t<S>::type;
					S val = std::bit_cast<S>(~U(0));
					return logic_xor(a, val);
				}
			};
		}
	}
}