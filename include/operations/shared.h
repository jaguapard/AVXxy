#pragma once
#include "../namespace.h"
#include "../FeatureSet.h"
#include "../SIMD_Vector.h"
#include "../SIMD_Mask.h"
#include "../meta/meta.h"
#include <source_location>

namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		using namespace meta;
		struct OperationBase
		{
		public:
			static inline constexpr internals::FeatureSet FS = internals::FS_current;
		protected:
			struct null_t {};

			struct avx512_vmbi2 {};
			struct avx512_vmbi {};
			struct avx512_dq {};
			struct avx512_vl {};
			struct avx512_bw {};
			struct avx512_cd {};
			struct avx512_f {};
			struct f16c {};
			struct avx2 {};
			struct avx {};
			struct scalar {};

			using order = std::tuple<
				std::conditional_t<FS.has(AVX512_VBMI2), avx512_vmbi2, null_t>,
				std::conditional_t<FS.has(AVX512_VBMI), avx512_vmbi, null_t>,
				std::conditional_t<FS.has(AVX512_DQ), avx512_dq, null_t>,
				std::conditional_t<FS.has(AVX512_VL), avx512_vl, null_t>,
				std::conditional_t<FS.has(AVX512_BW), avx512_bw, null_t>,
				std::conditional_t<FS.has(AVX512_CD), avx512_cd, null_t>,
				std::conditional_t<FS.has(AVX512_F), avx512_f, null_t>,
				std::conditional_t<FS.has(F16C), f16c, null_t>,
				std::conditional_t<FS.has(AVX2), avx2, null_t>,
				std::conditional_t<FS.has(AVX), avx, null_t>,
				scalar
			>;
		public:
			//Maximum vector size of bytes, or 0 if no vector operations are supported
			//Value of 0 can be here only if scalar operations supported
			static inline constexpr size_t MaxVectorSize = []() {
				if constexpr (FS.has(AVX512_F)) return 64;
				else if constexpr (FS.has(AVX)) return 32;
				else if constexpr (FS.has(SSE)) return 16;
				else return 0;
				}();
			/*
			template<size_t VecSizeThreshold, typename... Args>
			static auto split_trivial(Args&&... args)
			{
				if constexpr (requires {split_trivial(std::forward<Args>(args)...); }
			}*/

			template<typename... Args>
			static auto eval(Args&&... args)
			{
				//prevent outside callers from messing up I
				eval_private(std::forward<Args>(args)...);
			}

		protected:
			template<size_t I = 0, typename... Args>
			static auto eval_private(Args&&... args)
			{
				if constexpr (I < std::tuple_size_v<order>)
				{
					using instr_set = std::tuple_element_t<I, order>;
					//Search tuple for fitting implementation, and return the value returned by first valid implementation.
					//If no implementations exist, static_assert triggers
					if constexpr (requires {run<instr_set>(std::forward<Args>(args)...); })
					{
						auto ret = run<instr_set>(std::forward<Args>(args)...);
						//if null_t is returned, it means that implementation exists, but it all fell through to the null_t return,
						//This is considered invalid, so continue searching
						if constexpr (std::is_same_v<decltype(ret), null_t>) return eval_private<I + 1>(std::forward<Args>(args)...);
						else return ret;
					}
					else return eval_private<I + 1>(std::forward<Args>(args)...);
				}
				else static_assert(always_false_v<Args...>, "AVXxy routing: no implementation exists for operation");
			}
		public:
			//scream your lungs out if scalar fallback is reached and this function is enabled via AVXXY_NOISY_SCALAR define
			static void scream(std::source_location loc = std::source_location::current())
			{
#ifdef AVXXY_NOISY_SCALAR
				std::cout << "\nScalar fallback reached:" << loc.function_name() << "\n";
#endif
			}

			template<typename S, size_t N, bool Lo>
			static SIMD_Vector<S, N> scalar_unpack_base(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
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
		};
	}
}