#pragma once
#include "namespace.h"

namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		template <typename To> struct op_cvt {};
		struct op_add {};
		struct op_sub {};
		struct op_mul {};
		struct op_div {};

	}
}