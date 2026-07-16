#pragma once
#include "../settings.h"
#include <type_traits>
#include <bit>

namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		template<typename To, typename From>
		requires (std::is_trivially_copyable_v<From> && std::is_trivially_copyable_v<To>)
		//Reinterprets input `a` as any other type `To`.
		//If sizeof(a) > sizeof(To), the upper bytes of `a` are discarded
		//If sizeof(a) < sizeof(To), then upper bytes of return value are undefined (if ZERO_FILL_UNSAFES setting is false) or set to zero (if ZERO_FILL_UNSAFES is true)
		//No out of bounds accesses are made in all cases, the callers are not required to check input and output sizes
		To avxxy_bit_cast(const From& a)
		{
			if constexpr (sizeof(From) == sizeof(To)) return std::bit_cast<To>(a);
			else
			{
				To ret;
				if constexpr (settings::ZERO_FILL_UNSAFES && sizeof(ret) > sizeof(a)) memset(&ret, 0, sizeof(ret));
				memcpy(&ret, &a, std::min(sizeof(ret), sizeof(a)));
				return ret;
			}
		}
	}
}
