#pragma once
#include "../namespace.h"
#include "enums.h"
#include <type_traits>
#include "concepts.h"

namespace AVXXY_NAMESPACE
{
	namespace meta
	{
		template<class...>
		inline constexpr bool always_false_v = false;

		static constexpr VectorSizeClassEnum vector_size_class(size_t sizeInBytes)
		{
			if (sizeInBytes <= 16) return VectorSizeClassEnum::XMM;
			else if (sizeInBytes <= 32) return VectorSizeClassEnum::YMM;
			else if (sizeInBytes <= 64) return VectorSizeClassEnum::ZMM;
			else return VectorSizeClassEnum::XL;
		}

		template<typename T> //requires (SupportsSizeClass<T>) 
		inline constexpr VectorSizeClassEnum vector_size_class_v = vector_size_class(sizeof(T));

		template<typename T> requires (IsScalarType<T>)
		inline constexpr ScalarSizeClassEnum scalar_size_class_v = []() {
			if constexpr (sizeof(T) == 1) return ScalarSizeClassEnum::byte;
			else if constexpr (sizeof(T) == 2) return ScalarSizeClassEnum::word;
			else if constexpr (sizeof(T) == 4) return ScalarSizeClassEnum::dword;
			else return ScalarSizeClassEnum::qword;
		}();

		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_fp16 = std::is_same_v<T, fp16_t>;
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_bf16 = std::is_same_v<T, bf16_t>;
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_f32 = std::is_same_v<T, float>;
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_f64 = std::is_same_v<T, double>;
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_i64 = std::is_same_v<T, int64_t>;
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_i32 = std::is_same_v<T, int32_t>;
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_i16 = std::is_same_v<T, int16_t>;
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_i8 = std::is_same_v<T, int8_t>;
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_u64 = std::is_same_v<T, uint64_t>;
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_u32 = std::is_same_v<T, uint32_t>;
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_u16 = std::is_same_v<T, uint16_t>;
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_u8 = std::is_same_v<T, uint8_t>;
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_small_sint = is_i16<T> || is_i8<T>;
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_small_uint = is_u16<T> || is_u8<T>;
		template <typename T> requires (IsScalarType<T>) inline constexpr bool any_small_int = is_small_sint<T> || is_small_uint<T>;
		template <typename T> requires (IsScalarType<T>) inline constexpr bool any_float = is_any_of_v<T, float, double, fp16_t, bf16_t>;
		//indicates wheter this type is 8 bit integer, signed or unsigned
		template <typename T> inline constexpr bool any_i8 = (is_u8<T> || is_i8<T>);
		//indicates wheter this type is 16 bit integer, signed or unsigned
		template <typename T> inline constexpr bool any_i16 = (is_u16<T> || is_i16<T>);
		//indicates wheter this type is 32 bit integer, signed or unsigned
		template <typename T> inline constexpr bool any_i32 = (is_u32<T> || is_i32<T>);
		//indicates wheter this type is 64 bit integer, signed or unsigned
		template <typename T> inline constexpr bool any_i64 = (is_u64<T> || is_i64<T>);
		//indicates wheter this type is integral
		template <typename T> requires (IsScalarType<T>) inline constexpr bool any_int = std::is_integral_v<T>;
		//indicates wheter this type is not integral (TODO: limit it only to doubles and floats, and maybe FP16/BF16?)
		template <typename T> requires (IsScalarType<T>) inline constexpr bool not_int = !std::is_integral_v<T>;
	}
}
