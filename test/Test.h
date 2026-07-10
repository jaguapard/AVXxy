#pragma once
#include "../include/avxxy.h"
#include "oracles.h"
#include <random>
#include <vector>
#include <span>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include "Reporter.h"

using namespace AVXXY_NAMESPACE;
using namespace meta;

class OneTimeMessageManager
{
private:
	std::ofstream file = std::ofstream("test_logs/general.log");
	std::ostringstream buf;
	std::unordered_set<std::string> seenMessages;
public:
	struct SendOff {};
	static constexpr SendOff sendoff;

	template<typename T>
	OneTimeMessageManager& operator<<(const T& v)
	{
		if constexpr (std::same_as<T, SendOff>)
		{
			std::string s = buf.str();
			if (seenMessages.insert(s).second)
			{
				std::cout << s;
				file << s;
			}
			buf.str("");
		}
		else buf << v;
		return *this;
	}
};
class Test
{
private:
	Reporter reporter;
	std::mt19937_64 mt = std::mt19937_64(30'06'2026);

	//Fills all the inputs with uniformly distributed bits
	template<typename... T>
	void fillUniform(T&... outs)
	{
		auto fillUniformOne = [&](auto& x) {
			static_assert(std::is_trivially_copyable_v<std::remove_reference<decltype(x)>>);

			std::uniform_int_distribution<int64_t> distr(INT64_MIN, INT64_MAX);
			std::byte* p = reinterpret_cast<std::byte*>(&x);
			constexpr size_t sz = sizeof(x);
			int64_t g;
			for (size_t i = 0; i < sz; i += sizeof(g))
			{
				g = distr(this->mt);
				memcpy(p + i, &g, std::min(sizeof(g), sz - i));
			}
			};
		(fillUniformOne(outs), ...);
	}
	//template<typename FilterFunc, typename... T>
	//void fillUniformFiltered

	template<typename CandRetT, typename RefRetT, typename... Args>
	void compareAndReport(const std::source_location& loc, const char* testName, const CandRetT& candRet, const RefRetT& refRet, Args... _args)
	{
		static_assert(sizeof(candRet) == sizeof(refRet));
		if (memcmp(&candRet, &refRet, sizeof(candRet)) != 0) [[unlikely]]
		{
			ReporterInput repInp(candRet, refRet, testName);
			repInp.loc = loc;
			auto appendReporter = [&](const auto& arg) {
				using ArgT = std::remove_const_t<std::remove_reference_t<decltype(arg)>>;
				static_assert(std::is_trivially_copyable_v<ArgT>);
				repInp.args.emplace_back(arg);
				};
			
			(appendReporter(_args), ...);
			this->reporter.reportFailure(repInp);
		}
	}

	template<typename... Args>
	void outputTestSkipMessage(std::source_location loc, const char* testName, Args... _args)
	{
		onetime << "Test " << testName << " at " << loc.function_name() << ", line " << loc.line() << ":\n";
		onetime << "Skipped due to it being illegal with these arguments:\n";
		size_t i = 0;
		auto dumpOne = [&](const auto& arg) {
			using ArgT = std::remove_const_t<std::remove_reference_t<decltype(arg)>>;
			onetime << "Arg " << i++ << ": " << typeid(ArgT).name() << "\n";
		};
		(dumpOne(_args), ...);
		onetime << "\n" << onetime.sendoff;
	}

	using type_order = std::tuple<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>;//, float, double>;//, fp16_t, bf16_t>;
	static constexpr std::array<size_t, 7> vector_sizes = { 1,2,4,8,16,32,64 };

	OneTimeMessageManager onetime;
	std::unordered_map<std::string, size_t> runsPerTest;
public:
	Test()
	{
		for (size_t i = 0; i < 16; ++i)
		{
			test_inner<0, 0>();
			test_combo<0, 0, 0>();
		}
		size_t neverRanTestCount = 0;
		for (auto& it : runsPerTest)
		{
			//If a test was registered, but never ran, it may suggest that it was always rejected as failing requirements.
			//Failing requirements occasionally is fine (operations may be illegal for some combinations of types),
			//But failing all of them is never correct, thus, warn about it.
			if (it.second == 0)
			{
				neverRanTestCount++;
				onetime << "Test " << it.first << " has never been ran!\n" << onetime.sendoff;
			}
		}
		if (neverRanTestCount != 0)
		{
			this->~Test(); //yes, like this. Need to close files and flush them, and I'm too lazy now to do it properly.
			throw std::runtime_error("Never ran some tests. Check test_logs/general.log");
		}
	}
private:
#define TEST(op, ...) do {\
	runsPerTest[#op];\
	auto loc = std::source_location::current();\
	if constexpr (requires { op(__VA_ARGS__); Oracles::op(__VA_ARGS__);}) {\
		auto candRet = op(__VA_ARGS__);\
		auto refRet = Oracles::op(__VA_ARGS__);\
		runsPerTest[#op]++;\
		compareAndReport(loc, #op, candRet, refRet, __VA_ARGS__);\
	}\
	else outputTestSkipMessage(loc, #op, __VA_ARGS__);} while(0);

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshift-count-overflow"
	template<size_t N = 0, typename T>
	void test_immediate_shifts(const T& a)
	{
		TEST(shift_left<N>, a);
		TEST(shift_right<N>, a);
		if constexpr (N <= 65) test_immediate_shifts<N + 1>(a);
	}
#pragma clang diagnostic pop
	template<size_t TypeIndex, size_t SizeIndex>
	void test_inner()
	{
		constexpr size_t TypeCount = std::tuple_size_v<type_order>;
		if constexpr (TypeIndex >= TypeCount) return;
		else if constexpr (SizeIndex >= vector_sizes.size()) return test_inner<TypeIndex + 1, 0>();
		else //both current type and size are legal
		{
			using S = std::tuple_element_t<TypeIndex, type_order>;
			constexpr size_t N = vector_sizes[SizeIndex];
			using VectorT = SIMD_Vector<S, N>;

			{
				std::cout << "Testing " << typeid(VectorT).name() << "...\n";
				//for (size_t k = 0; k < 16; ++k)
				{
					SIMD_Vector<S, N> a, b, c;
					uint64_t maskBits;
					this->fillUniform(a, b, c, maskBits);

					mask_t<S, N> mask = maskBits;
					TEST(add, a, b);
					TEST(sub, a, b);
					TEST(mul, a, b);
					TEST(logic_and, a, b);
					TEST(logic_or, a, b);
					TEST(logic_xor, a, b);
					TEST(logic_not, a);
					this->test_immediate_shifts(a);
					
					TEST(min, a, b);
					TEST(max, a, b);
					TEST(sqrtf, a);
					TEST(sqrtd, a);

					TEST(maskz_mov, mask, a);
					TEST(mask_mov, a, mask, b);
					TEST(blend, mask, a, b);

					TEST(cmp_equal, a, b);
					TEST(cmp_not_equal, a, b);
					TEST(cmp_less, a, b);
					TEST(cmp_less_or_equal, a, b);
					TEST(cmp_greater, a, b);
					TEST(cmp_greater_or_equal, a, b);

					TEST(unpacklo, a, b);
					TEST(unpackhi, a, b);
					TEST(conflict, a);
					TEST(vpopcnt, a);
				}
			}
			test_inner<TypeIndex, SizeIndex + 1>();
		}
	}

	template<size_t TypeIndex1, size_t TypeIndex2, size_t SizeIndex>
	void test_combo()
	{
		constexpr size_t TypeCount = std::tuple_size_v<type_order>;
		if constexpr (TypeIndex1 >= TypeCount) return;
		else if constexpr (TypeIndex2 >= TypeCount) return test_combo<TypeIndex1 + 1, 0, 0>();
		else if constexpr (SizeIndex >= vector_sizes.size()) return test_combo<TypeIndex1, TypeIndex2+1, 0>();
		else
		{
			using S1 = std::tuple_element_t<TypeIndex1, type_order>;
			using S2 = std::tuple_element_t<TypeIndex2, type_order>;
			constexpr size_t N = vector_sizes[SizeIndex];
			{
				SIMD_Vector<S1, N> a, b;
				SIMD_Vector<S2, N> x, y;
				uint64_t maskBits;
				fillUniform(a, b, x, y, maskBits);
				mask_t<S1, N> mask = maskBits;

				TEST(vcvt<S2>, a);
				TEST(permx, a, x);
				TEST(permx2, a, b, x);
				TEST(shift_left, a, x & 127);
				TEST(shift_right, a, x & 127);
				TEST(vrzext<S2>, a);
				TEST(vrtrunc<S2>, a);
				TEST(byte_shuffle, a, b);
				TEST(vsat<S2>, a);
				//TEST(movm<S1>, mask);
			}
			return test_combo<TypeIndex1, TypeIndex2, SizeIndex + 1>();
		}
	}
};