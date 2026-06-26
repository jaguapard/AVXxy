#pragma once
#include "namespace.h"

namespace AVXXY_NAMESPACE
{
	namespace settings
	{
		static constexpr bool noisyScalar = false;

		//FP16 operations without native support (AVX512-FP16) are emulated via convert to FP32 + do operation + convert back to FP16
		//Repeated conversions introduce performance penalty and less precision in calculations
		//This setting governs wheter or not to skip back-conversion when returning the results of the mathematical operations
		//However, if FP16 conversion is required (i.e. assigning result to FP16 variable), the conversion will still be performed
		//This does not affect data-type-agnostic operations (data movement)
		//Default value is true		
		//TODO: this may cause major headaches on false, make wrapper class for FP32-in-FP16 clothing?
		//static constexpr bool FP16_emulation_returns_FP16_only = true;
	}
}