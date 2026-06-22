#pragma once
#include "../SIMD_Vector.h"
#include "../Dispatcher.h"
#include "../op_tags.h"
#include <source_location>

namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		struct ISA_Scalar
		{
			static inline constexpr FeatureSet FS = internals::FS_current;
			template<typename Op, typename S, size_t N>
			requires (std::same_as<Op, op_add>)
			static SIMD_Vector<S, N> eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i] + b[i];
				return ret;
			}

		private:
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