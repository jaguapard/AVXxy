#pragma once
#include "../SIMD_Vector.h"
#include "../Dispatcher.h"
#include "../op_tags.h"
#include <source_location>

#include "../SIMD_Mask.h"
#include "../SIMD_Vector.h"
#include "../FeatureSet.h"
#include <iostream>

namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		struct ISA_Scalar
		{
			static inline constexpr FeatureSet FS = internals::FS_current;
			template<typename Op, typename S, size_t N>
			requires (std::same_as<Op,op_add>)
			static SIMD_Vector<S, N> eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i] + b[i];
				return ret;
			}

			template<typename To, size_t N, typename From>
			static SIMD_Vector<To, N> eval(op_cvt<To>, const SIMD_Vector<From, N>& a)
			{
				scream();
				SIMD_Vector<To, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i];
				return ret;
			}
			/*
			template<typename S, size_t N>
				requires (sizeof(S) * 8 >= N)
			static SIMD_Vector<typename same_size_uint_t<S>::type, N> eval(op_conflict, const SIMD_Vector<S, N>& a)
			{
				using U = same_size_uint_t<S>::type;
				using T = SIMD_Vector<U, N>;
				T ret;
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

			template<typename S, size_t N>
			static SIMD_Vector<S, N> eval(op_sub, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i] - b[i];
				return ret;
			}
			template<typename S, size_t N>
			static SIMD_Vector<S, N> eval(op_mul, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i] * b[i];
				return ret;
			}
			template<typename S, size_t N>
			static SIMD_Vector<S, N> eval(op_div, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i] / b[i];
				return ret;
			}
			template<typename S, size_t N>
			static SIMD_Vector<S, N> eval(op_mod, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i)
					if constexpr (concepts::any_int<S>) ret[i] = a[i] % b[i];
					else ret[i] = std::fmod(a[i], b[i]); //TODO: should this even exist?
				return ret;
			}


			//TODO: limit bitwise operations to int types?
			template<typename S, size_t N>
			static SIMD_Vector<S, N> eval(op_or, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				using T = typename concepts::same_size_uint_t<S>::type;
				for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(std::bit_cast<T>(a[i]) | std::bit_cast<T>(b[i]));
				return ret;
			}
			template<typename S, size_t N>
			static SIMD_Vector<S, N> eval(op_and, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				using T = typename concepts::same_size_uint_t<S>::type;
				for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(std::bit_cast<T>(a[i]) & std::bit_cast<T>(b[i]));
				return ret;
			}
			template<typename S, size_t N>
			static SIMD_Vector<S, N> eval(op_xor, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				using T = typename concepts::same_size_uint_t<S>::type;
				for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(S(std::bit_cast<T>(a[i]) ^ std::bit_cast<T>(b[i])));
				return ret;
			}
			template<typename S, size_t N>
			static SIMD_Vector<S, N> eval(op_not, const SIMD_Vector<S, N>& a)
			{
				scream();
				SIMD_Vector<S, N> ret;
				using T = typename concepts::same_size_uint_t<S>::type;
				for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(~std::bit_cast<T>(a[i]));
				return ret;
			}


			template<typename S, size_t N, typename I>
				requires (concepts::any_int<S>&& concepts::any_int<I>)
			static SIMD_Vector<S, N> eval(op_shl, const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				using T = typename concepts::same_size_uint_t<S>::type;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i] << b[i];
				return ret;
			}
			template<typename S, size_t N, typename I>
				requires (concepts::any_int<S>&& concepts::any_int<I>)
			static SIMD_Vector<S, N> eval(op_shr, const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				using T = typename concepts::same_size_uint_t<S>::type;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i] >> b[i];
				return ret;
			}

			template<typename S, size_t N>
			static SIMD_Vector<float, N> eval(op_sqrtf, const SIMD_Vector<S, N>& a)
			{
				scream();
				SIMD_Vector<float, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = std::sqrt(float(a[i]));
				return ret;
			}
			template<typename S, size_t N>
			static SIMD_Vector<double, N> eval(op_sqrtd, const SIMD_Vector<S, N>& a)
			{
				scream();
				SIMD_Vector<double, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = std::sqrt(double(a[i]));
				return ret;
			}

			template<typename S, size_t N, typename I>
				requires (concepts::any_int<I>)
			static SIMD_Vector<S, N> eval(op_permx, const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = a[ind[i] & (N - 1)];
				return ret;
			}
			template<typename S, size_t N, typename I>
				requires (concepts::any_int<I>)
			static SIMD_Vector<S, N> eval(op_permx2, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const SIMD_Vector<I, N>& ind)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i)
				{
					auto j = ind[i] & (2 * N - 1);
					ret[i] = j < N ? a[j] : b[j - N];
				}
				return ret;
			}

			template<typename S, size_t N>
				requires (std::is_floating_point_v<S>)
			static SIMD_Vector<S, N> eval(op_floor, const SIMD_Vector<S, N>& a)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = std::floor(a[i]);
				return ret;
			}
			template<typename S, size_t N>
				requires (std::is_floating_point_v<S>)
			static SIMD_Vector<S, N> eval(op_ceil, const SIMD_Vector<S, N>& a)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = std::ceil(a[i]);
				return ret;
			}


			template<typename S, size_t N>
			static SIMD_Vector<S, N> eval(op_load<S, N>, const void* p, const typename SIMD_Vector<S, N>::MaskT& mask, const SIMD_Vector<S, N>& src)
			{
				scream();
				SIMD_Vector<S, N> ret;
				const S* sp = static_cast<const S*>(p);
				for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? sp[i] : src[i];
				return ret;
			}
			template<typename S, size_t N>
			static void eval(op_store, SIMD_Vector<S, N> vec, void* p, const typename SIMD_Vector<S, N>::MaskT& mask)
			{
				scream();
				S* sp = static_cast<S*>(p);
				for (size_t i = 0; i < N; ++i) if (mask[i]) sp[i] = vec[i];
			}
			template<typename S, size_t N, size_t Scale, typename I>
				requires (concepts::any_int<I>)
			static SIMD_Vector<S, N> eval(op_gather<S, N, Scale>, const void* base, const SIMD_Vector<I, N>& ind, const typename SIMD_Vector<S, N>::MaskT& mask, const SIMD_Vector<S, N>& src = 0)
			{
				scream();
				SIMD_Vector<S, N> ret;
				size_t addr = size_t(base);
				for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? *(const S*)(addr + Scale * ind[i]) : src[i];
				return ret;
			}
			template<typename S, size_t N, size_t Scale, typename I>
				requires (concepts::any_int<I>)
			static void eval(op_scatter<Scale>, const SIMD_Vector<S, N>& v, void* base, const SIMD_Vector<I, N>& ind, const typename SIMD_Vector<S, N>::MaskT& mask = typename SIMD_Vector<S, N>::MaskT::AllOnes())
			{
				scream();
				size_t addr = size_t(base);
				for (size_t i = 0; i < N; ++i) if (mask[i]) *(S*)(addr + Scale * ind[i]) = v[i];
			}


			template<typename S, size_t N>
			static SIMD_Vector<S, N> eval(op_compress, const typename SIMD_Vector<S, N>::MaskT& mask, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& src = 0)
			{
				scream();
				SIMD_Vector<S, N> ret;
				size_t j = 0;
				for (size_t i = 0; i < N; ++i) if (mask[i]) ret[j++] = a[i];
				for (; j < N; ++j) ret[j] = src[j];
				return ret;
			}

			template<typename S, size_t N>
			static SIMD_Vector<S, N> eval(op_mask_mov, const SIMD_Vector<S, N>& ifBitClear, const typename SIMD_Vector<S, N>::MaskT& mask, const SIMD_Vector<S, N>& ifBitSet)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? ifBitSet[i] : ifBitClear[i];
				return ret;
			}

			template<typename S, size_t N>
			static SIMD_Vector<S, N> eval(op_unpacklo, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				return unpack_base<S, N, true>(a, b);
			}
			template<typename S, size_t N>
			static SIMD_Vector<S, N> eval(op_unpackhi, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				return unpack_base<S, N, false>(a, b);
			}

			template<typename S, size_t N>
			static typename SIMD_Vector<S, N>::MaskT eval(op_cmpeq, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				typename SIMD_Vector<S, N>::MaskT ret = 0;
				for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] == b[i]);
				return ret;
			}
			template<typename S, size_t N>
			static typename SIMD_Vector<S, N>::MaskT eval(op_cmpneq, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				typename SIMD_Vector<S, N>::MaskT ret = 0;
				for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] != b[i]);
				return ret;
			}
			template<typename S, size_t N>
			static typename SIMD_Vector<S, N>::MaskT eval(op_cmplt, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				typename SIMD_Vector<S, N>::MaskT ret = 0;
				for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] < b[i]);
				return ret;
			}
			template<typename S, size_t N>
			static typename SIMD_Vector<S, N>::MaskT eval(op_cmple, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				typename SIMD_Vector<S, N>::MaskT ret = 0;
				for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] <= b[i]);
				return ret;
			}
			template<typename S, size_t N>
			static typename SIMD_Vector<S, N>::MaskT eval(op_cmpgt, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				typename SIMD_Vector<S, N>::MaskT ret = 0;
				for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] > b[i]);
				return ret;
			}
			template<typename S, size_t N>
			static typename SIMD_Vector<S, N>::MaskT eval(op_cmpge, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				typename SIMD_Vector<S, N>::MaskT ret = 0;
				for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] >= b[i]);
				return ret;
			}


			template<typename S, size_t N>
			static SIMD_Vector<S, N> eval(op_abs, const SIMD_Vector<S, N>& a)
			{
				scream();
				if constexpr (std::is_unsigned_v<S>) return a;
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = std::abs(a[i]);
				return ret;
			}
			template<typename S, size_t N>
			static SIMD_Vector<S, N> eval(op_min, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = std::min(a[i], b[i]);
				return ret;
			}
			template<typename S, size_t N>
			static SIMD_Vector<S, N> eval(op_max, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = std::max(a[i], b[i]);
				return ret;
			}

			template<typename S, size_t N>
			static SIMD_Vector<S, N> eval(op_movm<S, N>, const typename bits_to_uint_t<N>::type& mask)
			{
				using U = concepts::same_size_uint_t<S>::type;
				using MU = bits_to_uint_t<N>::type;
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i)
				{
					U x = mask & (MU(1) << i) ? ~U(0) : U(0);
					S y = std::bit_cast<S>(x);
					ret[i] = y;
				}
				return ret;
			}
			template<typename S, size_t N>
			static typename bits_to_uint_t<N>::type eval(op_movemask, const SIMD_Vector<S, N>& a)
			{
				using U = concepts::same_size_uint_t<S>::type;
				using X = typename bits_to_uint_t<N>::type;
				X ret = 0;
				constexpr U sb = U(1) << (sizeof(U) * 8 - 1);
				for (size_t i = 0; i < N; ++i) if (std::bit_cast<U>(a[i]) & sb) ret |= X(1) << i;
				return ret;
			}
		private:
			template<typename S, size_t N, bool Lo>
			static SIMD_Vector<S, N> unpack_base(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				SIMD_Vector<S, N> ret;
				constexpr size_t pairs_per_xmm = 8 / sizeof(S); //8, since unpack only processes lower/upper half of each input
				constexpr size_t elements_per_xmm = 16 / sizeof(S); //how much elements of type S fit into one 128 bit lane
				constexpr size_t xmm_count = sizeof(ret) / 16;
				for (size_t xmm_i = 0; xmm_i < xmm_count; ++xmm_i) //for each 128-bit lane
				{
					for (size_t i = 0; i < elements_per_xmm; i += 2)
					{
						size_t srcI = xmm_i * elements_per_xmm + i / 2 + (Lo ? 0 : elements_per_xmm / 2);
						ret[xmm_i * elements_per_xmm + i] = a[srcI];
						ret[xmm_i * elements_per_xmm + i + 1] = b[srcI];
					}
				}
				return ret;
			}
			*/
			//scream your lungs out if scalar fallback is reached and this function is enabled via AVXXY_NOISY_SCALAR define
			static void scream(std::source_location loc = std::source_location::current())
			{
#ifdef AVXXY_NOISY_SCALAR
				std::cout << "\nScalar fallback reached:" << loc.function_name() << "\n";
#endif
			}
		};
	}
}