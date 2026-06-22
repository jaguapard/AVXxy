#pragma once
#include "shared.h"
namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		struct op_conflict : OperationBase
		{
			template<typename S, size_t N>
				requires (sizeof(S) * 8 >= N)
			static SIMD_Vector<typename ScalarTraits<S>::UintT, N> run(const SIMD_Vector<S, N>& a)
			{
				using U = ScalarTraits<S>::UintT;
				using T = SIMD_Vector<U, N>;
				T ret;
				scream();
				for (size_t i = 0; i < N; ++i)
				{
					U acc = 0;
					for (size_t j = 0; j < i; ++j)
					{
						if (a[i] == a[j]) acc |= U(1) << j;
					}
					ret[i] = acc;
				}
				return ret;
			}
		};

		struct op_div : OperationBase
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> run(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i] / b[i];
				return ret;
			}
		};

		struct op_mask_mov : OperationBase
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> run(const SIMD_Vector<S, N>& ifBitClear, const mask_t<S,N>& mask, const SIMD_Vector<S, N>& ifBitSet)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? ifBitSet[i] : ifBitClear[i];
				return ret;
			}
		};

		struct op_unpacklo : OperationBase
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> run(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				return scalar_unpack_base<S, N, true>(a, b);
			}
		};

		struct op_unpackhi : OperationBase
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> run(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				return scalar_unpack_base<S, N, false>(a, b);
			}
		};

		struct op_gather : OperationBase
		{
			template<typename S, size_t N, size_t Scale, typename I>
				requires (meta::any_int<I>)
			static SIMD_Vector<S, N> run(const void* base, const SIMD_Vector<I, N>& ind, const typename SIMD_Vector<S, N>::MaskT& mask, const SIMD_Vector<S, N>& src = 0)
			{
				scream();
				SIMD_Vector<S, N> ret;
				size_t addr = size_t(base);
				for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? *(const S*)(addr + Scale * ind[i]) : src[i];
				return ret;
			}
		};

		struct op_load : OperationBase
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> run(const void* p, const mask_t<S, N>& mask, const SIMD_Vector<S, N>& src)
			{
				scream();
				SIMD_Vector<S, N> ret;
				const S* sp = static_cast<const S*>(p);
				for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? sp[i] : src[i];
				return ret;
			}
		};

		struct op_max : OperationBase
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> run(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = std::max(a[i], b[i]);
				return ret;
			}
		};
		struct op_min : OperationBase
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> run(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = std::min(a[i], b[i]);
				return ret;
			}
		};
		struct op_mul : OperationBase
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> run(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i] * b[i];
				return ret;
			}
		};
		struct op_not : OperationBase
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> run(const SIMD_Vector<S, N>& a)
			{
				scream();
				SIMD_Vector<S, N> ret;
				using T = meta::ScalarTraits<S>::UintT;
				for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(~std::bit_cast<T>(a[i]));
				return ret;
			}
		};
	}
}