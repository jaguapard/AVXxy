#pragma once
#include "../include/avxxy.h"

using namespace AVXXY_NAMESPACE;
using namespace meta;
using namespace internals;

//Oracle implementations for operations. DO NOT use it in real code, it's only for testing!
struct Oracles
{
private:
	template<typename S, size_t N, bool Lo>
	static SIMD_Vector<S, N> unpack_base(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		SIMD_Vector<S, N> ret;
		constexpr size_t pairs_per_xmm = 8 / sizeof(S); //8, since unpack only processes lower/upper half of each input
		constexpr size_t elements_per_xmm = 16 / sizeof(S); //how much elements of type S fit into one 128 bit lane
		constexpr size_t xmm_count = sizeof(ret) / 16 + bool(sizeof(ret) % 16);
		for (size_t xmm_i = 0; xmm_i < xmm_count; ++xmm_i) //for each 128-bit lane
		{
			for (size_t i = 0; i < elements_per_xmm; i += 2)
			{
				size_t srcI = xmm_i * elements_per_xmm + i / 2 + (Lo ? 0 : elements_per_xmm / 2);
				size_t i1 = xmm_i * elements_per_xmm + i;
				size_t i2 = i1 + 1;
				if (i1 < N) ret[i1] = a[srcI];
				if (i2 < N) ret[i2] = b[srcI];
			}
		}
		return ret;
	}
public:
	//Performs element-wise addition of vectors and returns the result
	template<typename S, size_t N> static SIMD_Vector<S, N> add(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = a[i] + b[i];
		return ret;
	}
	//Performs element-wise subtraction of elements of vector b from elements of vector a and returns the result
	template<typename S, size_t N> static SIMD_Vector<S, N> sub(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = a[i] - b[i];
		return ret;
	}
	//Multiplies elements of input vectors together and returns the result. For integer vectors, only lower half of the 2x-sized intermediate result is returned
	template<typename S, size_t N> static SIMD_Vector<S, N> mul(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = a[i] * b[i];
		return ret;
	}

	//Performs divison of two vectors and returns the result
	template<typename S, size_t N> static SIMD_Vector<S, N> div(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = a[i] / b[i];
		return ret;
	}



	//Returns bitwise logical and of the two vectors. Floating point vectors are also legibile for this operation.
	template<typename S, size_t N> static SIMD_Vector<S, N> logic_and(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using U = ScalarTraits<S>::UintT;
		if constexpr (!any_int<S>) return Oracles::vcast<S>(Oracles::logic_and(Oracles::vcast<U>(a), Oracles::vcast<U>(b)));
		else
		{
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = a[i] & b[i];
			return ret;
		}
	}

	//Returns bitwise logical or of the two vectors. Floating point vectors are also legibile for this operation.
	template<typename S, size_t N> static SIMD_Vector<S, N> logic_or(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using U = ScalarTraits<S>::UintT;
		if constexpr (!any_int<S>) return Oracles::vcast<S>(Oracles::logic_or(Oracles::vcast<U>(a), Oracles::vcast<U>(b)));
		else
		{
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = a[i] | b[i];
			return ret;
		}
	}

	//Returns bitwise logical exclusive or of the two vectors. Floating point vectors are also legibile for this operation.
	template<typename S, size_t N> static SIMD_Vector<S, N> logic_xor(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using U = ScalarTraits<S>::UintT;
		if constexpr (!any_int<S>) return Oracles::vcast<S>(Oracles::logic_xor(Oracles::vcast<U>(a), Oracles::vcast<U>(b)));
		else
		{
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = a[i] ^ b[i];
			return ret;
		}
	}

	//Returns bitwise logical negation of the input vector. Floating point vectors are also legibile for this operation.
	template<typename S, size_t N> static SIMD_Vector<S, N> logic_not(const SIMD_Vector<S, N>& a)
	{
		using U = ScalarTraits<S>::UintT;
		if constexpr (!any_int<S>) return Oracles::vcast<S>(Oracles::logic_not(Oracles::vcast<U>(a)));
		else
		{
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = ~a[i];
			return ret;
		}
	}



	//Shift packed integers in `a` left by the amount specified by the corresponding element of `amount` while shifting in zeros, and returns the result.
	//If the shift amount is greater or equal to number of bits in S, the value is set to zero
	//Amount is treated as unsigned integer
	template<meta::any_int S, size_t N, meta::any_int I>
	static SIMD_Vector<S, N> shift_left(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& amount)
	{
		using canon_t = meta::ScalarTraits<S>::UintT;
		using UI = meta::ScalarTraits<I>::UintT;
		if constexpr (!std::same_as<I, canon_t>) return Oracles::shift_left(a, Oracles::vsat<canon_t>(vcvt<UI>(amount)));
		else
		{
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i)
			{
				if (amount[i] < sizeof(S) * 8) ret[i] = a[i] << amount[i];
				else ret[i] = 0;
			}
			return ret;
		}
	}

	//Shift packed integers in `a` left by the amount specified by the template parameter A while shifting in zeros, and returns the result.
	//If the shift amount is greater or equal to number of bits in S, the value is set to zero
	template<size_t A, meta::any_int S, size_t N>
	static SIMD_Vector<S, N> shift_left(const SIMD_Vector<S, N>& a)
	{
		if constexpr (A >= sizeof(S) * 8) return 0;
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = a[i] << A;
		return ret;
	}

	//Shift packed integers in `a` right by the amount specified by the corresponding element of `amount` while shifting in sign bits, and returns the result.
	//If the shift amount is greater or equal to number of bits in S, the corresponding lane is set to zero (if `a` is unsigned) or broadcasted sign bit (if `a` is signed) 
	//Amount is treated as unsigned integer
	template<meta::any_int S, size_t N, meta::any_int I>
	static SIMD_Vector<S, N> shift_right(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& amount)
	{
		using canon_t = meta::ScalarTraits<S>::UintT;
		using UI = meta::ScalarTraits<I>::UintT;
		if constexpr (!std::same_as<I, canon_t>) return Oracles::shift_right(a, Oracles::vsat<canon_t>(vcvt<UI>(amount)));
		else
		{
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i)
			{
				if (amount[i] < sizeof(S) * 8) ret[i] = a[i] >> amount[i];
				else ret[i] = a[i] < 0 ? meta::AllOnes<S> : meta::AllZeros<S>;
			}
			return ret;
		}
	}

	//Shift packed integers in `a` right by the amount specified by the template parameter A while shifting in sign bits, and returns the result.
	template<size_t A, meta::any_int S, size_t N>
	static SIMD_Vector<S, N> shift_right(const SIMD_Vector<S, N>& a)
	{
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i)
		{
			if (A < sizeof(S) * 8) ret[i] = a[i] >> A;
			else ret[i] = a[i] < 0 ? meta::AllOnes<S> : meta::AllZeros<S>;
		}
		return ret;
	}

	//Converts integral input to other integral vector by using saturation and returns the result.
	//The input is clamped to output's scalar type range
	template<meta::any_int To, meta::any_int From, size_t N>
	static SIMD_Vector<To, N> vsat(const SIMD_Vector<From, N>& a)
	{
		SIMD_Vector<To, N> ret;
		for (size_t i = 0; i < N; ++i)
		{
			if (a[i] > std::numeric_limits<To>::max()) ret[i] = std::numeric_limits<To>::max();
			else if (a[i] < std::numeric_limits<To>::min()) ret[i] = std::numeric_limits<To>::min();
			else ret[i] = a[i];
		}
		return ret;
	}


	//Performs permutation on the elements from vector `a`. Elements of the returned vector are gathered from vector `a` by indices passed in `ind`.
	//Indices outside the range [0, N-1] wrap around N (-1 maps to N-1, N maps to 0).
	//ret[i] = a[ind[i] & (N-1)]
	template<typename S, size_t N, meta::any_int I> static SIMD_Vector<S, N> permx(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind)
	{
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i)
		{
			auto t = ind[i] % N;
			if (t < 0) t += N;
			ret[i] = a[t];
		}
		return ret;
	}

	//Appends vector `b` to vector `a`, then performs permutation on the elements from this temporary value. 
	//Elements of the returned vector are gathered from temporary vector by indices passed in `ind` 
	//Indices outside the range [0, 2*N-1] wrap around 2*N (-1 maps to 2*N-1, 2*N maps to 0).
	//t = ind[i] & (2*N - 1)
	//ret[i] = t < N ? a[t] : b[t-N]
	template<typename S, size_t N, meta::any_int I> static SIMD_Vector<S, N> permx2(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const SIMD_Vector<I, N>& ind)
	{
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i)
		{
			auto t = ind[i] % (2 * N);
			if (t < 0) t += 2 * N;
			ret[i] = t < N ? a[t] : b[t - N];
		}
		return ret;
	}



	//Converts the input to single-precision floating point numbers, then returns the square root of this value
	template<typename S, size_t N> static SIMD_Vector<float, N> sqrtf(const SIMD_Vector<S, N>& a)
	{
		SIMD_Vector<float, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = std::sqrt(float(a[i]));
		return ret;
	}
	//Converts the input to double-precision floating point numbers, then returns the square root of this value
	template<typename S, size_t N> static SIMD_Vector<double, N> sqrtd(const SIMD_Vector<S, N>& a)
	{
		SIMD_Vector<double, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = std::sqrt(double(a[i]));
		return ret;
	}

	//Converts the vector of one scalar type to vector of another scalar type and returns the result
	//For floating point to integer conversions, the input vector is truncated
	//For integer to bigger integer conversions, the input vector is sign or zero extended, depending on input signedness
	//For integer to smaller integer conversions, the input vector is wrapped around small integer's max value (TODO: is it true?)
	template<meta::IsScalarType To, size_t N, meta::IsScalarType From> static SIMD_Vector<To, N> vcvt(const SIMD_Vector<From, N>& value)
	{
		SIMD_Vector<To, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = value[i];
		return ret;
	}

	//Concatenates vectors in order they are passed to function call (left to right) and returns the result.
	//Leftmost vector is copied to lowest bits of the output, then second leftmost is appended to it, etc
	//Until the final rightmost vector that is copied to the highest bits of the output
	//The resultant vector's size is equal to sum of all input sizes
	template<typename S, size_t... Ns>
	static auto concat(const SIMD_Vector<S, Ns>&... vectors);

	//vrzext - vector reinterpret and zero-extend
	//Zero-extends each element of input vector to sizeof(S2) bytes and returns the resultant vector reinterpreted to output type
	//requires output scalar type to be larger or equal in size to input scalar type
	//i.e. vrzext<double>(SIMD_Vector<int8_t, 8>>) will put input values in lowest byte of each lane in the returned vector, while upper 7 bytes of each lane are filled with zeros
	//If input and output scalar sizes match, the input vector is only reinterpreted as output vector
	template<typename S2, typename S, size_t N>
		requires (sizeof(S2) >= sizeof(S))
	static SIMD_Vector<S2, N> vrzext(const SIMD_Vector<S, N>& a)
	{
		SIMD_Vector<S2, N> ret;
		memset(&ret, 0, sizeof(ret));

		for (size_t i = 0; i < N; ++i) memcpy(&ret[i], &a[i], sizeof(S));
		return ret;
	}

	//vrtrunc - vector reinterpret and truncate
	//Discards upper sizeof(S)-sizeof(S2) bytes from each element in the input vector and returns the resultant vector reinterpreted to vector of S2.
	//requires output scalar type to be less or equal in size to input scalar type
	//i.e. vrtrunc<int16_t>(SIMD_Vector<double, 8>) will discard upper 6 bytes each input double.
	//If input and output scalar sizes match, the input vector is only reinterpreted as output vector
	template<typename S2, typename S, size_t N>
		requires (sizeof(S2) <= sizeof(S))
	static SIMD_Vector<S2, N> vrtrunc(const SIMD_Vector<S, N>& a)
	{
		SIMD_Vector<S2, N> ret;
		for (size_t i = 0; i < N; ++i) memcpy(&ret[i], &a[i], sizeof(S2));
		return ret;
	}

	//Reinterprets input vector as any non-scalar type of any size
	//Requires the output type to be trivially copyable
	//If T is a scalar type, the vector is reinterpreted as vector of other scalar type
	//with lane count calculated automatically to be smallest vector that is bigger or the same size as input
	//i.e. vcast<uint32_t>(SIMD_Vector<uint8_t, 3>) will return SIMD_Vector<uint32_t, 1>
	//If output type is larger than input, the upper bytes of output are undefined
	//If output type is smaller than input, the upper bytes of input are discarded
	template<typename To, typename S, size_t N> requires (std::is_trivially_copyable_v<To>)
		static auto vcast(const SIMD_Vector<S, N>& a)
	{
		using T = SIMD_Vector<S, N>;
		if constexpr (meta::IsScalarType<To>)
		{
			constexpr size_t RetN = sizeof(T) / sizeof(To) + bool(sizeof(T) % sizeof(To));
			SIMD_Vector<To, RetN> ret;
			memcpy(&ret, &a, std::min(sizeof(ret), sizeof(a)));
			return ret;
		}
		else
		{
			To ret;
			memcpy(&ret, &a, std::min(sizeof(ret), sizeof(a)));
			return ret;
		}
	}

	//Selects elements from two input vectors by corresponding mask bits and returns the result.
	//If the mask bit is 0, the corresponding element of `ifBitClear` is chosen
	//If the mask bit is 1, the corresponding element of `ifBitSet` is chosen
	//This function differs from blend only by the order of it's arguments
	//ret[i] = mask[i] ? ifBitSet[i] : ifBitClear[i]
	template <typename S, size_t N> static SIMD_Vector<S, N> mask_mov(const SIMD_Vector<S, N>& ifBitClear, const mask_t<S, N>& mask, const SIMD_Vector<S, N>& ifBitSet)
	{
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? ifBitSet[i] : ifBitClear[i];
		return ret;
	}
	//Selects elements from input vector by corresponding mask bits and returns the result.
	//If the mask bit is 0, the corresponding element of the returned vector is set to zero
	//If the mask bit is 1, the corresponding element of `ifBitSet` is chosen
	//ret[i] = mask[i] ? ifBitSet[i] : S(0)
	template <typename S, size_t N> static SIMD_Vector<S, N> maskz_mov(const mask_t<S, N>& mask, const SIMD_Vector<S, N>& ifBitSet)
	{
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? ifBitSet[i] : 0;
		return ret;
	}
	//Selects elements from two input vectors by corresponding mask bits and returns the result.
	//If the mask bit is 0, the corresponding element of `ifBitClear` is chosen
	//If the mask bit is 1, the corresponding element of `ifBitSet` is chosen
	//This function differs from mask_mov only by the order of it's arguments
	//ret[i] = mask[i] ? ifBitSet[i] : ifBitClear[i]
	template <typename S, size_t N> static SIMD_Vector<S, N> blend(const mask_t<S, N>& mask, const SIMD_Vector<S, N>& ifBitClear, const SIMD_Vector<S, N>& ifBitSet)
	{
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? ifBitSet[i] : ifBitClear[i];
		return ret;
	}

	//Loads vector from memory p and returns the result. The memory does not have to be aligned
	template<typename S, size_t N> static SIMD_Vector<S, N> load(const void* p)
	{
		SIMD_Vector<S, N> ret;
		memcpy(&ret, p, sizeof(ret));
		return ret;
	}

	//Loads vector from memory p and returns the result. The memory does not have to be aligned
	template<meta::IsSimdVector T> T __forceinline load(const void* p)
	{
		return load<typename T::ScalarT, T::LaneCount>(p);
	}

	//Loads vector from aligned memory p and returns the result. The pointer p must be aligned to boundary depending on output size:
	//16 bytes for vectors less than or equal to 16 bytes
	//32 bytes for vectors sized between 17 and 32 bytes inclusive
	//64 bytes for vectors larger than 32 bytes
	template<typename S, size_t N> static SIMD_Vector<S, N> load_a(const void* p)
	{
		SIMD_Vector<S, N> ret;
		memcpy(&ret, p, sizeof(ret));
		return ret;
	}

	//Loads vector from aligned memory p and returns the result. The pointer p must be aligned to boundary depending on output size:
	//16 bytes for vectors less than or equal to 16 bytes
	//32 bytes for vectors sized between 17 and 32 bytes inclusive
	//64 bytes for vectors larger than 32 bytes
	template<meta::IsSimdVector T> T __forceinline load_a(const void* p)
	{
		return load_a<typename T::ScalarT, T::LaneCount>(p);
	}

	//Loads the vector from unaligned memory location pointed to by `p` and returns the result.
	//If the corresponding mask bit is set, the corresponding element in memory is read and stored into the returned vector
	//If the corresponding mask bit is cleared, the corresponding element in memory is not read and the destination element is zeroed out
	//Masked out elements are guaranteed to not cause memory-related faults
	//ret[i] = mask[i] ? reinterpret_cast<const S*>(p)[i] : std::bit_cast<S>(0);
	template<typename S, size_t N> static SIMD_Vector<S, N> load(const void* p, const mask_t<S, N>& mask)
	{
		SIMD_Vector<S, N> ret;
		const S* sp = reinterpret_cast<const S*>(p);
		for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? sp[i] : 0;
		return ret;
	}
	template<meta::IsSimdVector T> static T __forceinline load(const void* p, const typename T::MaskT& mask)
	{
		return load<typename T::ScalarT, T::LaneCount>(p, mask);
	}

	//Loads the vector from unaligned memory location pointed to by `p` and returns the result.
	//If the corresponding mask bit is set, the corresponding element in memory is read and stored into the returned vector
	//If the corresponding mask bit is cleared, the corresponding element in memory is not read and the corresponding element from src is stored into the retuned vector
	//Masked out elements are guaranteed to not cause memory-related faults
	//ret[i] = mask[i] ? reinterpret_cast<const S*>(p)[i] : src[i]
	template<typename S, size_t N> static SIMD_Vector<S, N> load(const void* p, const mask_t<S, N>& mask, const SIMD_Vector<S, N>& src)
	{
		SIMD_Vector<S, N> ret;
		const S* sp = reinterpret_cast<const S*>(p);
		for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? sp[i] : src[i];
		return ret;
	}
	//Loads the vector from unaligned memory location pointed to by `p` and returns the result.
	//If the corresponding mask bit is set, the corresponding element in memory is read and stored into the returned vector
	//If the corresponding mask bit is cleared, the corresponding element in memory is not read and the corresponding element from src is stored into the retuned vector
	//Masked out elements are guaranteed to not cause memory-related faults
	//ret[i] = mask[i] ? reinterpret_cast<const S*>(p)[i] : src[i]
	template<meta::IsSimdVector T>
	static __forceinline T load(const void* p, const typename T::MaskT& mask, const T& src)
	{
		return load<typename T::ScalarT, T::LaneCount>(p, mask, src);
	}

	//Conditionally stores vector `v` to memory location pointed by `p` using mask `mask`.
	//If the corresponding mask bit is set, the corresponding element of `v` is stored into the memory
	//Else, no action is performed
	//Masked out elements are guaranteed to not cause memory-related faults
	//if (mask[i]) reinterpret_cast<S*>(p)[i] = v[i]
	template<typename S, size_t N> static void store(const SIMD_Vector<S, N>& v, void* p, const mask_t<S, N>& mask = mask_t<S, N>::AllOnesUint)
	{
		S* sp = reinterpret_cast<S*>(p);
		for (size_t i = 0; i < N; ++i) if (mask[i]) sp[i] = v[i];
	}

	//Conditionally gathers elements from memory, stores them into a vector and returns the result.
	//If the corresponding mask bit is set, the corresponding element in memory is read and stored into the returned vector
	//If the corresponding mask bit is cleared, the corresponding element in memory is not read and the corresponding element from src is stored into the retuned vector
	//Masked out elements are guaranteed to not cause memory-related faults
	//By default, scale is set to the size of vector's scalar type
	//ret[i] = mask[i] ? *reinterpret_cast<const S*>(size_t(base) + Scale*ind[i]) : src[i]
	template<typename S, size_t N, size_t Scale = sizeof(S), meta::any_int I>
	__forceinline static  SIMD_Vector<S, N> gather(const void* base, const SIMD_Vector<I, N>& ind, const mask_t<S, N>& mask = mask_t<S, N>::AllOnesUint, const SIMD_Vector<S, N>& src = 0)
	{
		return __gather_impl<S, N, Scale>(base, ind, mask, src);
	}

	//Conditionally gathers elements from memory, stores them into a vector and returns the result.
	//If the corresponding mask bit is set, the corresponding element in memory is read and stored into the returned vector
	//If the corresponding mask bit is cleared, the corresponding element in memory is not read and the corresponding element from src is stored into the retuned vector
	//Masked out elements are guaranteed to not cause memory-related faults
	//By default, scale is set to the size of vector's scalar type
	//ret[i] = mask[i] ? *reinterpret_cast<const S*>(size_t(base) + Scale*ind[i]) : src[i]
	template <meta::IsSimdVector T, size_t Scale = sizeof(typename T::ScalarT), meta::any_int I>
	__forceinline static T gather(const void* base, const SIMD_Vector<I, T::LaneCount>& ind, const typename T::MaskT& mask = T::MaskT::AllOnesUint, const T& src = 0)
	{
		return __gather_impl<typename T::ScalarT, T::LaneCount, Scale>(base, ind, mask, src);
	}

	//Conditionally scatters vector `v` to memory location pointed by `base` using mask `mask`.
	//If the corresponding mask bit is set, the corresponding element of `v` is stored into the memory
	//Else, no action is performed
	//Masked out elements are guaranteed to not cause memory-related faults
	//By default, scale is set to the size of vector's scalar type
	//if (mask[i]) *reinterpret_cast<S*>(size_t(base) + Scale*ind[i]) = v[i]
	template<typename S, size_t N, size_t Scale = sizeof(S), meta::any_int I>
	static void scatter(const SIMD_Vector<S, N>& vec, void* base, const SIMD_Vector<I, N>& ind, const mask_t<S, N>& mask = mask_t<S, N>::AllOnesUint)
	{
		for (size_t i = 0; i < N; ++i) if (mask[i]) *reinterpret_cast<S*>(size_t(base) + Scale * ind[i]) = vec[i];
	}

	//Performs the element-wise comparsion and returns the resultant mask.
	//If elements are equal, the corresponding mask bit is set to 1
	//Otherwise, the corresponding mask bit is set to 0
	//ret[i] = a[i] == b[i]
	template<typename S, size_t N> static mask_t<S, N> cmp_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		mask_t<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] == b[i]);
		return ret;
	}
	//Performs the element-wise comparsion and returns the resultant mask.
	//If elements are not equal, the corresponding mask bit is set to 1
	//Otherwise, the corresponding mask bit is set to 0
	//ret[i] = a[i] != b[i]
	template<typename S, size_t N> static mask_t<S, N> cmp_not_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		mask_t<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] != b[i]);
		return ret;
	}
	//Performs the element-wise comparsion and returns the resultant mask.
	//If element of vector `a` is less than element of vector `b`, the corresponding mask bit is set to 1
	//Otherwise, the corresponding mask bit is set to 0
	//ret[i] = a[i] < b[i]
	template<typename S, size_t N> static  mask_t<S, N> cmp_less(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		mask_t<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] < b[i]);
		return ret;
	}
	//Performs the element-wise comparsion and returns the resultant mask.
	//If element of vector `a` is less than or equal to element of vector `b`, the corresponding mask bit is set to 1
	//Otherwise, the corresponding mask bit is set to 0
	//ret[i] = a[i] <= b[i]
	template<typename S, size_t N> static mask_t<S, N> cmp_less_or_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		mask_t<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] <= b[i]);
		return ret;
	}
	//Performs the element-wise comparsion and returns the resultant mask.
	//If element of vector `a` is greater than element of vector `b`, the corresponding mask bit is set to 1
	//Otherwise, the corresponding mask bit is set to 0
	//ret[i] = a[i] > b[i]
	template<typename S, size_t N> static mask_t<S, N> cmp_greater(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		mask_t<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] > b[i]);
		return ret;
	}
	//Performs the element-wise comparsion and returns the resultant mask.
	//If element of vector `a` is greater than or equal to element of vector `b`, the corresponding mask bit is set to 1
	//Otherwise, the corresponding mask bit is set to 0
	//ret[i] = a[i] >= b[i]
	template<typename S, size_t N> static mask_t<S, N> cmp_greater_or_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		mask_t<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] >= b[i]);
		return ret;
	}

	//Returns absolute value of the input vector
	//The returned values are undefined for signed elements equal to their minimum value
	template<typename S, size_t N> static SIMD_Vector<S, N> abs(const SIMD_Vector<S, N>& a)
	{
		if constexpr (std::is_unsigned_v<S>) return a;
		else
		{
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = a[i] >= 0 ? a[i] : -a[i];// fuck this stupid ambigious garbage, really std::abs(S(a[i]));
			return ret;
		}
	}
	//Rounds each element of input vector towards negative infinity (floor) and returns the result.
	template<meta::any_float S, size_t N> static SIMD_Vector<S, N> floor(const SIMD_Vector<S, N>& a)
	{
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = std::floor(a[i]);
		return ret;
	}
	//Rounds each element of input vector towards positive infinity (ceil) and returns the result.
	template<meta::any_float S, size_t N> static SIMD_Vector<S, N> ceil(const SIMD_Vector<S, N>& a)
	{
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = std::ceil(a[i]);
		return ret;
	}
	//Compares two vectors together element-wise and returns the lower ones.
	//ret[i] = a[i] < b[i] ? a[i] : b[i]
	template<typename S, size_t N> static SIMD_Vector<S, N> min(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = std::min(a[i], b[i]);
		return ret;
	}
	//Compares two vectors together element-wise and returns the higher ones.
	//ret[i] = a[i] > b[i] ? a[i] : b[i]
	template<typename S, size_t N> static SIMD_Vector<S, N> max(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = std::max(a[i], b[i]);
		return ret;
	}
	//Compares vector `val` to vectors `min` and `max`
	//If corresponding value in `val` is outside the range [min..max], it will be forced to min or max.
	//tmp = val[i] < min[i] ? min[i] : val[i]
	//ret[i] = tmp > max[i] ? max[i] : tmp
	template<typename S, size_t N> static SIMD_Vector<S, N> clamp(const SIMD_Vector<S, N>& val, const SIMD_Vector<S, N>& min, const SIMD_Vector<S, N>& max)
	{
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = std::clamp(val[i], min[i], max[i]);
		return ret;
	}

	//Split input vectors into 128-bit chunks. Upper halves of each chunk are discarded.
	//For each result chunk, even elements are picked from a, while odd elements are picked from b.
	//Chunks are merged back into the resultant vector in the same order they appear in input vectors
	//chunk_ret[i] = i % 2 == 0 ? chunk_a[i/2] : chunk_b[i/2]
	template<typename S, size_t N> static SIMD_Vector<S, N> unpacklo(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return Oracles::unpack_base<S, N, true>(a, b);
	}
	//Split input vectors into 128-bit chunks. Lower halves of each chunk are discarded.
	//For each result chunk, even elements are picked from a, while odd elements are picked from b.
	//Chunks are merged back into the resultant vector in the same order they appear in input vectors
	//x = 8 bytes / sizeof(S)
	//chunk_ret[i] = i % 2 == 0 ? chunk_a[x+i/2] : chunk_b[x+i/2]
	template<typename S, size_t N> 
		requires meta::unpackhi_legal<S,N>
	static SIMD_Vector<S, N> unpackhi(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return Oracles::unpack_base<S, N, false>(a, b);
	}

	//Copies vector `src` and conditionally overwrites it with elements of vector `a`
	//Mask is iterated from lower bits to higher ones. 
	//If the mask bit is set, the corresponding element is read from `a` and is written to return vector at pivot point, 
	//advancing pivot point is by one element. Otherwise, no action is performed.
	//ret = src; pivot = 0
	//for (size_t i = 0; i < N; ++i)
	//    if (mask[i]) ret[pivot++] = a[i];
	template <typename S, size_t N> static SIMD_Vector<S, N> compress(const mask_t<S, N>& mask, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& src = 0)
	{
		SIMD_Vector<S, N> ret;
		size_t j = 0;
		for (size_t i = 0; i < N; ++i) if (mask[i]) ret[j++] = a[i];
		for (; j < N; ++j) ret[j] = src[j];
		return ret;
	}

	//Iterates input vector from lowest elements to highest
	//For each element, checks elements below it, and sets corresponding output element's bit to 1 if it's equal to the tested element, or 0 otherwise
	//This function is only available for vectors in which elements have number of bits greater or equal to vector's lane count
	//ret = 0;
	//for (size_t i = 0; i < N; ++i)
	//    for (size_t j = 0; j < i; ++j)
	//        if (a[i] == a[j]) ret[i] |= 1 << j; 
	template <typename S, size_t N> requires (sizeof(S) * 8 >= N)
		static SIMD_Vector<typename meta::ScalarTraits<S>::UintT, N> conflict(const SIMD_Vector<S, N>& a)
	{
		using U = ScalarTraits<S>::UintT;
		using UV = SIMD_Vector<U, N>;
		UV ret;
		ret[0] = 0;
		for (size_t i = 1; i < N; ++i)
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

	//For each element in `a`, computes the number of set bits and stores the computed value into corresponding element of returned vector
	//for (size_t i = 0; i < N; ++i) ret[i] = popcnt(a[i])
	template<meta::vpopcnt_allowed S, size_t N> static SIMD_Vector<typename meta::ScalarTraits<S>::UintT, N> vpopcnt(const SIMD_Vector<S, N>& a)
	{
		using U = ScalarTraits<S>::UintT;
		using UV = SIMD_Vector<U, N>;
		UV ret;

		for (size_t i = 0; i < N; ++i) ret[i] = std::popcount(std::bit_cast<U>(a[i]));
		return ret;
	}

	//Extracts uppermost bit of each element and returns them as mask.
	template <typename S, size_t N> static mask_t<S, N> movemask(const SIMD_Vector<S, N>& v)
	{
		mask_t<S, N> ret = 0;
		using I = ScalarTraits<S>::IntT;
		for (size_t i = 0; i < N; ++i) ret.setBit(std::bit_cast<I>(v[i]) < 0);
		return ret;
	}
	//Sets all bits of each element to 0 if corresponding mask bit is 0, or 1 otherwise.
	//@tparam S scalar type of the returned vector
	//@tparam N number of lanes in returned vector, same as bit count of input mask
	//@tparam C size class of the input mask
	template <typename S, meta::ScalarSizeClassEnum C, size_t N> static SIMD_Vector<S, N> movm(const internals::SIMD_Mask<C, N>& mask)
	{
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(mask[i] ? ScalarTraits<S>::AllOnesUint : 0);
		return ret;
	}

	//Reinterprets `a` as vector of bytes, then shuffles these bytes within 128-bit lanes by indices `b`.
	//After the shuffle is done, reinterprets the shuffled vector back to input type and returns it.
	//Only uppermost bit and 4 lowest bits of each index byte are used for the shuffle.
	//If uppermost bit of the index in `b` is set, then corresponding output lane is zeroed out.
	//Otherwise, the byte is taken from the same 128-bit lane of `a` by index b[i] & 15.
	//X = sizeof(a)
	//for (size_t start = 0; start < X; start += 16)
	//    for (size_t i = 0; i < std::min(X-start, 16); ++i)
	//        ret[start + i] = b[start + i] > 127 ? 0 : a[start + (b[start+i] & 15)]
	template<typename S, size_t N>
	static SIMD_Vector<S, N> byte_shuffle(const SIMD_Vector<S, N>& a, const SIMD_Vector<uint8_t, N * sizeof(S)>& b)
	{
		if constexpr (!is_u8<S>) return Oracles::vcast<S>(Oracles::byte_shuffle(Oracles::vcast<uint8_t>(a), b));
		else
		{
			//TODO: breaks for small vectors!
			SIMD_Vector<S, N> ret;
			constexpr size_t X = sizeof(a);
			for (size_t start = 0; start < X; start += 16)
			{
				for (size_t i = 0; i < std::min<size_t>(X - start, 16); ++i)
				{
					auto y = b[start + i];
					ret[start + i] = y > 127 ? 0 : a[start + (y & 15)];
				}
			}
			return ret;
		}
	}


	//Performs a block permutation of input vector by compile-time-known indices.
	//Requires size of input to be divisible by size of block.
	//Requires number of indices and number of blocks in input to match.
	//Requires all indices to be in range 0 to C-1 inclusive, where C in number of blocks in the input vector.
	//@tparam Block This type's size is used as permutation granularity. Only scalar and vector types are accepted
	//@tparam Idx zero-indexed source block indices. Output block i is copied from input block Idx[i]
	template<typename Block, size_t... Idx, typename S, size_t N>
	static SIMD_Vector<S, N> permute(const SIMD_Vector<S, N>& a);

	//Performs a block permutation of 2 input vectors by compile-time-known indices.
	//Requires size of inputs to be divisible by size of block.
	//Requires number of indices and number of blocks in input to match.
	//Requires all indices to be in range 0 to 2*C-1 inclusive, where C in number of blocks in the input vector.
	//@tparam Block This type's size is used as permutation granularity. Only scalar and vector types are accepted
	//@tparam Idx zero-indexed source block indices. Output block i is copied from a's block Idx[i] if index is less than C or from b's block Idx[i] otherwise 
	template<typename Block, size_t... Idx, typename S, size_t N>
	static SIMD_Vector<S, N> permute2(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
};