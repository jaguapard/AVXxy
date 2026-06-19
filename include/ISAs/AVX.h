#pragma once
#include "../namespace.h"
#include "../tags.h"
#include "../SIMD_BitMask.h"
#include "../SIMD_Vector.h"
#include "../FeatureSet.h"
#include "../funcs.h"
#include "../tables.h"

namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		namespace ISA
		{
			using namespace concepts;
			using namespace utils;
			template<internals::FeatureSet FS>
			struct AVX
			{
				template<typename S, size_t N>
					requires ((is_f32<S> || is_f64<S>) && sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_abs, const SIMD_Vector<S, N>& a)
				{
					using U = same_size_uint_t<S>::type;
					constexpr U sb = U(1) << (sizeof(S) * 8 - 1);
					return logic_xor(a, std::bit_cast<S>(~sb)); //force sign bit to 0
				}
			};
		}
	}
}