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
			template<internals::FeatureSet FS>
			struct AVX512VL
			{
				template<typename S, size_t N>
					requires (FS.has(AVX512_F) && any_i64<S>)
				static SIMD_Vector<S, N> eval(op_abs, const SIMD_Vector<S, N>& a)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (ymm_sized<T>) return _mm256_abs_epi64(a);
					else if constexpr (xmm_sized<T>) return _mm_abs_epi64(a);
					else static_assert(always_false_v<S>);
				}

				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) <= 32 && ((FS.has(AVX512_F) && !any_small_int<S>) || (FS.has(AVX512_BW) && any_small_int<S>)))
				static SIMD_BitMask<N> eval(op_cmpeq, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (ymm_sized<T>)
					{
						if constexpr (FS.has(AVX512_F) && is_f64<S>) return _mm256_cmp_pd_mask(a, b, _CMP_EQ_OQ);
						else if constexpr (FS.has(AVX512_F) && is_f32<S>) return _mm256_cmp_ps_mask(a, b, _CMP_EQ_OQ);
						else if constexpr (FS.has(AVX512_F) && is_i64<S>) return _mm256_cmpeq_epi64_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_u64<S>) return _mm256_cmpeq_epu64_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_i32<S>) return _mm256_cmpeq_epi32_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_u32<S>) return _mm256_cmpeq_epu32_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm256_cmpeq_epi16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm256_cmpeq_epu16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm256_cmpeq_epi8_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm256_cmpeq_epu8_mask(a, b);
						else static_assert(always_false_v<S>);
					}
					else if constexpr (xmm_sized<T>)
					{
						if constexpr (FS.has(AVX512_F) && is_f64<S>) return _mm_cmp_pd_mask(a, b, _CMP_EQ_OQ);
						else if constexpr (FS.has(AVX512_F) && is_f32<S>) return _mm_cmp_ps_mask(a, b, _CMP_EQ_OQ);
						else if constexpr (FS.has(AVX512_F) && is_i64<S>) return _mm_cmpeq_epi64_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_u64<S>) return _mm_cmpeq_epu64_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_i32<S>) return _mm_cmpeq_epi32_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_u32<S>) return _mm_cmpeq_epu32_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm_cmpeq_epi16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm_cmpeq_epu16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm_cmpeq_epi8_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm_cmpeq_epu8_mask(a, b);
						else static_assert(always_false_v<S>);
					}
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) <= 32 && ((FS.has(AVX512_F) && !any_small_int<S>) || (FS.has(AVX512_BW) && any_small_int<S>)))
				static SIMD_BitMask<N> eval(op_cmpneq, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (ymm_sized<T>)
					{
						if constexpr (FS.has(AVX512_F) && is_f64<S>) return _mm256_cmp_pd_mask(a, b, _CMP_NEQ_OQ);
						else if constexpr (FS.has(AVX512_F) && is_f32<S>) return _mm256_cmp_ps_mask(a, b, _CMP_NEQ_OQ);
						else if constexpr (FS.has(AVX512_F) && is_i64<S>) return _mm256_cmpneq_epi64_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_u64<S>) return _mm256_cmpneq_epu64_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_i32<S>) return _mm256_cmpneq_epi32_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_u32<S>) return _mm256_cmpneq_epu32_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm256_cmpneq_epi16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm256_cmpneq_epu16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm256_cmpneq_epi8_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm256_cmpneq_epu8_mask(a, b);
						else static_assert(always_false_v<S>);
					}
					else if constexpr (xmm_sized<T>)
					{
						if constexpr (FS.has(AVX512_F) && is_f64<S>) return _mm_cmp_pd_mask(a, b, _CMP_NEQ_OQ);
						else if constexpr (FS.has(AVX512_F) && is_f32<S>) return _mm_cmp_ps_mask(a, b, _CMP_NEQ_OQ);
						else if constexpr (FS.has(AVX512_F) && is_i64<S>) return _mm_cmpneq_epi64_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_u64<S>) return _mm_cmpneq_epu64_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_i32<S>) return _mm_cmpneq_epi32_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_u32<S>) return _mm_cmpneq_epu32_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm_cmpneq_epi16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm_cmpneq_epu16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm_cmpneq_epi8_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm_cmpneq_epu8_mask(a, b);
						else static_assert(always_false_v<S>);
					}
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) <= 32 && ((FS.has(AVX512_F) && !any_small_int<S>) || (FS.has(AVX512_BW) && any_small_int<S>)))
				static SIMD_BitMask<N> eval(op_cmpgt, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (ymm_sized<T>)
					{
						if constexpr (FS.has(AVX512_F) && is_f64<S>) return _mm256_cmp_pd_mask(a, b, _CMP_GT_OQ);
						else if constexpr (FS.has(AVX512_F) && is_f32<S>) return _mm256_cmp_ps_mask(a, b, _CMP_GT_OQ);
						else if constexpr (FS.has(AVX512_F) && is_i64<S>) return _mm256_cmpgt_epi64_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_u64<S>) return _mm256_cmpgt_epu64_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_i32<S>) return _mm256_cmpgt_epi32_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_u32<S>) return _mm256_cmpgt_epu32_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm256_cmpgt_epi16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm256_cmpgt_epu16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm256_cmpgt_epi8_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm256_cmpgt_epu8_mask(a, b);
						else static_assert(always_false_v<S>);
					}
					else if constexpr (xmm_sized<T>)
					{
						if constexpr (FS.has(AVX512_F) && is_f64<S>) return _mm_cmp_pd_mask(a, b, _CMP_GT_OQ);
						else if constexpr (FS.has(AVX512_F) && is_f32<S>) return _mm_cmp_ps_mask(a, b, _CMP_GT_OQ);
						else if constexpr (FS.has(AVX512_F) && is_i64<S>) return _mm_cmpgt_epi64_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_u64<S>) return _mm_cmpgt_epu64_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_i32<S>) return _mm_cmpgt_epi32_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_u32<S>) return _mm_cmpgt_epu32_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm_cmpgt_epi16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm_cmpgt_epu16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm_cmpgt_epi8_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm_cmpgt_epu8_mask(a, b);
						else static_assert(always_false_v<S>);
					}
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) <= 32 && ((FS.has(AVX512_F) && !any_small_int<S>) || (FS.has(AVX512_BW) && any_small_int<S>)))
				static SIMD_BitMask<N> eval(op_cmpge, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (ymm_sized<T>)
					{
						if constexpr (FS.has(AVX512_F) && is_f64<S>) return _mm256_cmp_pd_mask(a, b, _CMP_GE_OQ);
						else if constexpr (FS.has(AVX512_F) && is_f32<S>) return _mm256_cmp_ps_mask(a, b, _CMP_GE_OQ);
						else if constexpr (FS.has(AVX512_F) && is_i64<S>) return _mm256_cmpge_epi64_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_u64<S>) return _mm256_cmpge_epu64_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_i32<S>) return _mm256_cmpge_epi32_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_u32<S>) return _mm256_cmpge_epu32_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm256_cmpge_epi16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm256_cmpge_epu16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm256_cmpge_epi8_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm256_cmpge_epu8_mask(a, b);
						else static_assert(always_false_v<S>);
					}
					else if constexpr (xmm_sized<T>)
					{
						if constexpr (FS.has(AVX512_F) && is_f64<S>) return _mm_cmp_pd_mask(a, b, _CMP_GE_OQ);
						else if constexpr (FS.has(AVX512_F) && is_f32<S>) return _mm_cmp_ps_mask(a, b, _CMP_GE_OQ);
						else if constexpr (FS.has(AVX512_F) && is_i64<S>) return _mm_cmpge_epi64_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_u64<S>) return _mm_cmpge_epu64_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_i32<S>) return _mm_cmpge_epi32_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_u32<S>) return _mm_cmpge_epu32_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm_cmpge_epi16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm_cmpge_epu16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm_cmpge_epi8_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm_cmpge_epu8_mask(a, b);
						else static_assert(always_false_v<S>);
					}
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) <= 32 && ((FS.has(AVX512_F) && !any_small_int<S>) || (FS.has(AVX512_BW) && any_small_int<S>)))
				static SIMD_BitMask<N> eval(op_cmplt, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (ymm_sized<T>)
					{
						if constexpr (FS.has(AVX512_F) && is_f64<S>) return _mm256_cmp_pd_mask(a, b, _CMP_LT_OQ);
						else if constexpr (FS.has(AVX512_F) && is_f32<S>) return _mm256_cmp_ps_mask(a, b, _CMP_LT_OQ);
						else if constexpr (FS.has(AVX512_F) && is_i64<S>) return _mm256_cmplt_epi64_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_u64<S>) return _mm256_cmplt_epu64_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_i32<S>) return _mm256_cmplt_epi32_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_u32<S>) return _mm256_cmplt_epu32_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm256_cmplt_epi16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm256_cmplt_epu16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm256_cmplt_epi8_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm256_cmplt_epu8_mask(a, b);
						else static_assert(always_false_v<S>);
					}
					else if constexpr (xmm_sized<T>)
					{
						if constexpr (FS.has(AVX512_F) && is_f64<S>) return _mm_cmp_pd_mask(a, b, _CMP_LT_OQ);
						else if constexpr (FS.has(AVX512_F) && is_f32<S>) return _mm_cmp_ps_mask(a, b, _CMP_LT_OQ);
						else if constexpr (FS.has(AVX512_F) && is_i64<S>) return _mm_cmplt_epi64_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_u64<S>) return _mm_cmplt_epu64_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_i32<S>) return _mm_cmplt_epi32_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_u32<S>) return _mm_cmplt_epu32_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm_cmplt_epi16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm_cmplt_epu16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm_cmplt_epi8_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm_cmplt_epu8_mask(a, b);
						else static_assert(always_false_v<S>);
					}
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) <= 32 && ((FS.has(AVX512_F) && !any_small_int<S>) || (FS.has(AVX512_BW) && any_small_int<S>)))
				static SIMD_BitMask<N> eval(op_cmple, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (ymm_sized<T>)
					{
						if constexpr (FS.has(AVX512_F) && is_f64<S>) return _mm256_cmp_pd_mask(a, b, _CMP_LE_OQ);
						else if constexpr (FS.has(AVX512_F) && is_f32<S>) return _mm256_cmp_ps_mask(a, b, _CMP_LE_OQ);
						else if constexpr (FS.has(AVX512_F) && is_i64<S>) return _mm256_cmple_epi64_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_u64<S>) return _mm256_cmple_epu64_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_i32<S>) return _mm256_cmple_epi32_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_u32<S>) return _mm256_cmple_epu32_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm256_cmple_epi16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm256_cmple_epu16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm256_cmple_epi8_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm256_cmple_epu8_mask(a, b);
						else static_assert(always_false_v<S>);
					}
					else if constexpr (xmm_sized<T>)
					{
						if constexpr (FS.has(AVX512_F) && is_f64<S>) return _mm_cmp_pd_mask(a, b, _CMP_LE_OQ);
						else if constexpr (FS.has(AVX512_F) && is_f32<S>) return _mm_cmp_ps_mask(a, b, _CMP_LE_OQ);
						else if constexpr (FS.has(AVX512_F) && is_i64<S>) return _mm_cmple_epi64_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_u64<S>) return _mm_cmple_epu64_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_i32<S>) return _mm_cmple_epi32_mask(a, b);
						else if constexpr (FS.has(AVX512_F) && is_u32<S>) return _mm_cmple_epu32_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm_cmple_epi16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm_cmple_epu16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm_cmple_epi8_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm_cmple_epu8_mask(a, b);
						else static_assert(always_false_v<S>);
					}
					else static_assert(always_false_v<S>);
				}

				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_mask_mov, const SIMD_Vector<S, N>& ifBitClear, const SIMD_BitMask<N>& mask, const SIMD_Vector<S, N>& ifBitSet)
					requires (sizeof(S) < 4 && FS.has(AVX512_BW) && (concepts::xmm_sized<SIMD_Vector<S, N>> || concepts::ymm_sized<SIMD_Vector<S, N>>))
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (ymm_sized<T> && any_i16<S>) return _mm256_mask_mov_epi16(ifBitClear, mask, ifBitSet);
					else if constexpr (ymm_sized<T> && any_i8<S>) return _mm256_mask_mov_epi8(ifBitClear, mask, ifBitSet);
					else if constexpr (xmm_sized<T> && any_i16<S>) return _mm_mask_mov_epi16(ifBitClear, mask, ifBitSet);
					else if constexpr (xmm_sized<T> && any_i8<S>) return _mm_mask_mov_epi8(ifBitClear, mask, ifBitSet);
					else static_assert(always_false_v<S>);
				}

				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_mask_mov, const SIMD_Vector<S, N>& ifBitClear, const SIMD_BitMask<N>& mask, const SIMD_Vector<S, N>& ifBitSet)
					requires (sizeof(S) >= 4 && FS.has(AVX512_F) && (concepts::xmm_sized<SIMD_Vector<S, N>> || concepts::ymm_sized<SIMD_Vector<S, N>>))
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (ymm_sized<T> && is_f64<S>) return _mm256_mask_mov_pd(ifBitClear, mask, ifBitSet);
					else if constexpr (ymm_sized<T> && is_f32<S>) return _mm256_mask_mov_ps(ifBitClear, mask, ifBitSet);
					else if constexpr (ymm_sized<T> && any_i64<S>) return _mm256_mask_mov_epi64(ifBitClear, mask, ifBitSet);
					else if constexpr (ymm_sized<T> && any_i32<S>) return _mm256_mask_mov_epi32(ifBitClear, mask, ifBitSet);

					else if constexpr (xmm_sized<T> && is_f64<S>) return _mm_mask_mov_pd(ifBitClear, mask, ifBitSet);
					else if constexpr (xmm_sized<T> && is_f32<S>) return _mm_mask_mov_ps(ifBitClear, mask, ifBitSet);
					else if constexpr (xmm_sized<T> && any_i64<S>) return _mm_mask_mov_epi64(ifBitClear, mask, ifBitSet);
					else if constexpr (xmm_sized<T> && any_i32<S>) return _mm_mask_mov_epi32(ifBitClear, mask, ifBitSet);
					else static_assert(always_false_v<S>);
				}

				template<typename S, size_t N, size_t Scale, typename I>
					requires (FS.has(AVX512_F) && concepts::any_int<I> && sizeof(S) >= 4 && std::max(sizeof(SIMD_Vector<S, N>), sizeof(SIMD_Vector<I, N>)) <= 32)
				static SIMD_Vector<S, N> eval(op_gather<S, N, Scale>, const void* base, const SIMD_Vector<I, N>& ind, const SIMD_BitMask<N>& mask = SIMD_BitMask<N>::AllOnes, const SIMD_Vector<S, N>& src = 0)
				{
					using namespace concepts;
					using CanonicalIndex_t = std::conditional_t<(sizeof(I) <= 4), int32_t, int64_t>;
					using RetVec_t = SIMD_Vector<S, N>;
					using IndVec_t = SIMD_Vector<I, N>;
					constexpr size_t MaxSize = std::max(sizeof(RetVec_t), sizeof(IndVec_t));
					//if scale is not native, emulate it by gathering with scale 1 and manually calculated byte offsets. 
					//TODO: Can optimize a little by checking if Scale*maxint(I) fits into smaller sizes
					if constexpr (Scale != 1 && Scale != 2 && Scale != 4 && Scale != 8) return gather<S, N, 1>(base, vcvt<int64_t>(ind) * Scale, mask, src);
					//TODO: emulation of small int gathers (where elements gathered are small ints)
					else if constexpr (!std::is_same_v<I, CanonicalIndex_t>) return gather<S, N, Scale>(base, vcvt<CanonicalIndex_t>(ind), mask, src);

					//if we get here, means that indices are already in good format (4-byte or 8-byte)
					else if constexpr (utils::is_ymm_size(MaxSize))
					{
						std::conditional_t<(concepts::ymm_sized<IndVec_t>), __m256i, __m128i> ni = ind;
						if constexpr (is_i64<I> && is_f64<S>) return _mm256_mmask_i64gather_pd(src, mask, ni, base, Scale);
						else if constexpr (is_i64<I> && is_f32<S>) return _mm256_mmask_i64gather_ps(src, mask, ni, base, Scale);
						else if constexpr (is_i64<I> && any_i64<S>) return _mm256_mmask_i64gather_epi64(src, mask, ni, base, Scale);
						else if constexpr (is_i64<I> && any_i32<S>) return _mm256_mmask_i64gather_epi32(src, mask, ni, base, Scale);

						else if constexpr (is_i32<I> && is_f64<S>) return _mm256_mmask_i32gather_pd(src, mask, ni, base, Scale);
						else if constexpr (is_i32<I> && is_f32<S>) return _mm256_mmask_i32gather_ps(src, mask, ni, base, Scale);
						else if constexpr (is_i32<I> && any_i64<S>) return _mm256_mmask_i32gather_epi64(src, mask, ni, base, Scale);
						else if constexpr (is_i32<I> && any_i32<S>) return _mm256_mmask_i32gather_epi32(src, mask, ni, base, Scale);
					}
					else if constexpr (utils::is_xmm_size(MaxSize))
					{
						__m128i ni = ind;
						if constexpr (is_i64<I> && is_f64<S>) return _mm_mmask_i64gather_pd(src, mask, ni, base, Scale);
						else if constexpr (is_i64<I> && is_f32<S>) return _mm_mmask_i64gather_ps(src, mask, ni, base, Scale);
						else if constexpr (is_i64<I> && any_i64<S>) return _mm_mmask_i64gather_epi64(src, mask, ni, base, Scale);
						else if constexpr (is_i64<I> && any_i32<S>) return _mm_mmask_i64gather_epi32(src, mask, ni, base, Scale);

						else if constexpr (is_i32<I> && is_f64<S>) return _mm_mmask_i32gather_pd(src, mask, ni, base, Scale);
						else if constexpr (is_i32<I> && is_f32<S>) return _mm_mmask_i32gather_ps(src, mask, ni, base, Scale);
						else if constexpr (is_i32<I> && any_i64<S>) return _mm_mmask_i32gather_epi64(src, mask, ni, base, Scale);
						else if constexpr (is_i32<I> && any_i32<S>) return _mm_mmask_i32gather_epi32(src, mask, ni, base, Scale);
					}
					else static_assert(always_false_v<I, S>);
				}

				template<typename S, size_t N, size_t Scale, typename I>
					requires (FS.has(AVX512_F) && concepts::any_int<I> && sizeof(S) >= 4 && std::max(sizeof(SIMD_Vector<S, N>), sizeof(SIMD_Vector<I, N>)) <= 32)
				static void eval(op_scatter<Scale>, const SIMD_Vector<S, N>& v, void* base, const SIMD_Vector<I, N>& ind, const SIMD_BitMask<N>& mask = SIMD_BitMask<N>::AllOnes)
				{
					using namespace concepts;
					//put everything up here to prevent else if chain breaks (since compilation gives useless errors by thinking unsanitized inputs surviving to native gathers
					using CanonicalIndex_t = std::conditional_t<(sizeof(I) <= 4), int32_t, int64_t>;
					using RetVec_t = SIMD_Vector<S, N>;
					using IndVec_t = SIMD_Vector<I, N>;
					constexpr size_t MaxSize = std::max(sizeof(RetVec_t), sizeof(IndVec_t));

					//if scale is not native, emulate it by gathering with scale 1 and manually calculated byte offsets. 
					//TODO: Can optimize a little by checking if Scale*maxint(I) fits into smaller sizes
					if constexpr (Scale != 1 && Scale != 2 && Scale != 4 && Scale != 8) return scatter<S, N, 1>(v, base, vcvt<int64_t>(ind) * Scale, mask);

					//TODO: emulation of small int scatter (where elements gathered are small ints)
					else if constexpr (!std::is_same_v<I, CanonicalIndex_t>) return scatter<S, N, Scale>(v, base, vcvt<CanonicalIndex_t>(ind), mask);

					else if constexpr (utils::is_ymm_size(MaxSize))
					{
						//clang is a cry-baby with ind here for some reason, so force convert it. Pay attention to size!
						std::conditional_t<(concepts::ymm_sized<IndVec_t>), __m256i, __m128i> ni = ind;
						if constexpr (is_i64<I> && is_f64<S>) return _mm256_mask_i64scatter_pd(base, mask, ni, v, Scale);
						else if constexpr (is_i64<I> && is_f32<S>) return _mm256_mask_i64scatter_ps(base, mask, ni, v, Scale);
						else if constexpr (is_i64<I> && any_i64<S>) return _mm256_mask_i64scatter_epi64(base, mask, ni, v, Scale);
						else if constexpr (is_i64<I> && any_i32<S>) return _mm256_mask_i64scatter_epi32(base, mask, ni, v, Scale);

						else if constexpr (is_i32<I> && is_f64<S>) return _mm256_mask_i32scatter_pd(base, mask, ni, v, Scale);
						else if constexpr (is_i32<I> && is_f32<S>) return _mm256_mask_i32scatter_ps(base, mask, ni, v, Scale);
						else if constexpr (is_i32<I> && any_i64<S>) return _mm256_mask_i32scatter_epi64(base, mask, ni, v, Scale);
						else if constexpr (is_i32<I> && any_i32<S>) return _mm256_mask_i32scatter_epi32(base, mask, ni, v, Scale);
						else static_assert(always_false_v<I, S>);
					}
					else if constexpr (utils::is_xmm_size(MaxSize))
					{
						//clang is a cry-baby with ind here for some reason, so force convert it. Pay attention to size!
						__m128i ni = ind;
						if constexpr (is_i64<I> && is_f64<S>) return _mm_mask_i64scatter_pd(base, mask, ni, v, Scale);
						else if constexpr (is_i64<I> && is_f32<S>) return _mm_mask_i64scatter_ps(base, mask, ni, v, Scale);
						else if constexpr (is_i64<I> && any_i64<S>) return _mm_mask_i64scatter_epi64(base, mask, ni, v, Scale);
						else if constexpr (is_i64<I> && any_i32<S>) return _mm_mask_i64scatter_epi32(base, mask, ni, v, Scale);

						else if constexpr (is_i32<I> && is_f64<S>) return _mm_mask_i32scatter_pd(base, mask, ni, v, Scale);
						else if constexpr (is_i32<I> && is_f32<S>) return _mm_mask_i32scatter_ps(base, mask, ni, v, Scale);
						else if constexpr (is_i32<I> && any_i64<S>) return _mm_mask_i32scatter_epi64(base, mask, ni, v, Scale);
						else if constexpr (is_i32<I> && any_i32<S>) return _mm_mask_i32scatter_epi32(base, mask, ni, v, Scale);
						else static_assert(always_false_v<I, S>);
					}
					else static_assert(always_false_v<I, S>);
				}
			};
		}
	}
}