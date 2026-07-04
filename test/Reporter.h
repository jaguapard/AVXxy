#pragma once
//#include "../include/avxxy.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <source_location>

enum class TypeEnum
{
	UNDEFINED,
	INT8,
	INT16,
	INT32,
	INT64,
	UINT64,
	UINT32,
	UINT16,
	UINT8,
	FLOAT,
	DOUBLE,
	FP16,
	BF16,	
};

struct ReporterArgInfo
{
	std::string argName, argTypeStr;
	TypeEnum elementType = TypeEnum::UNDEFINED;
	bool isPointer = false; //if true, the data hold pointer to elementType, not actual data
	std::vector<std::byte> data;
};

struct ReporterEntry
{
	const type_info* type;
	std::vector<std::byte> bytes;

	//ReporterEntry() {};

	template<typename T>
	ReporterEntry(const T& inp) : type(&typeid(T))
	{
		static_assert(std::is_trivially_copyable_v<T>);
		bytes.resize(sizeof(T));
		memcpy(bytes.data(), &inp, sizeof(T));
	}
};

struct ReporterInput
{
	std::vector<ReporterEntry> args;
	ReporterEntry refRet, candRet;
	std::string testName;
	std::source_location loc;

	template<typename T, typename U>
	ReporterInput(const T& candidate, const U& reference, const char* testName)://const std::string& testName) :
		refRet(reference), candRet(candidate), testName(testName ? testName : "(unnamed test)")
	{
	};
};

class Reporter
{
public:
	Reporter() {};
	void reportFailure(const ReporterInput& input);
private:
	std::unordered_map<std::string, std::ofstream> logFiles;
	static const std::unordered_map<TypeEnum, size_t> typeEnumToSize;
};