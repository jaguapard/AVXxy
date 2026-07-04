#include "Reporter.h"
#include <sstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <functional>
#include "../include/avxxy.h"
#include <typeindex>
#include "Table.h"

using namespace AVXXY_NAMESPACE;

const std::unordered_map<TypeEnum, size_t> Reporter::typeEnumToSize = []() {
	std::unordered_map<TypeEnum, size_t> ret;
	ret[TypeEnum::UNDEFINED] = 1;
	ret[TypeEnum::INT8] = 1;
	ret[TypeEnum::INT16] = 2;
	ret[TypeEnum::INT32] = 4;
	ret[TypeEnum::INT64] = 8;
	ret[TypeEnum::UINT8] = 1;
	ret[TypeEnum::UINT16] = 2;
	ret[TypeEnum::UINT32] = 4;
	ret[TypeEnum::UINT64] = 8;
	ret[TypeEnum::FLOAT] = 4;
	ret[TypeEnum::DOUBLE] = 8;
	ret[TypeEnum::FP16] = 2;
	ret[TypeEnum::BF16] = 2;
	return ret;
	}();

template<typename T>
T printDataToStreams(const std::byte* p, std::ostream& dec, std::ostream& hex)
{
	T ret;
	memset(&ret, 0, sizeof(ret));
	memcpy(&ret, p, sizeof(T));

	if (std::same_as<T, int8_t> || std::same_as<T, uint8_t>)
	{
		std::conditional_t<std::same_as<T, int8_t>, int32_t, uint32_t> ex = ret;
		dec << ex & 0xFF;
		hex << ex & 0xFF;
	}
	else
	{
		dec << ret;
		hex << ret;
	}
	return ret;
}

template<typename T>
void fillTableWithData(const void* start, size_t byteCount, Table& table, size_t hexCol, size_t decCol)
{
	std::ostringstream dec, hex;
	const T* p = reinterpret_cast<const T*>(start);
	size_t elementCount = byteCount / sizeof(T);
	if (byteCount % sizeof(T) != 0) throw std::runtime_error("non integral element count");

	hex << std::hex;

	for (size_t i = 0; i < elementCount; ++i)
	{
		T ret;
		//memset(&ret, 0, sizeof(ret));
		memcpy(&ret, p+i, sizeof(T));

		if (std::same_as<T, int8_t> || std::same_as<T, uint8_t>)
		{
			std::conditional_t<std::same_as<T, int8_t>, int32_t, uint32_t> ex = ret;
			dec << ex;
			hex << (ex & 0xFF);
		}
		else
		{
			dec << ret;
			hex << ret;
		}

		table.cellAt(i + 1, hexCol).value = hex.str();
		table.cellAt(i + 1, hexCol).extraData = ret;
		table.cellAt(i + 1, decCol).value = dec.str();
		table.cellAt(i + 1, decCol).extraData = ret;

		hex.str("");
		dec.str("");
	}
	//return ret;
}

using Printer = std::function<void(
	const void*,
	size_t,
	Table&,
	size_t,
	size_t
	)>;

struct TypeRegistryEntry
{
	//const type_info* type;
	Printer printFunc;
	size_t elemCount;
};

static const std::unordered_map<size_t, TypeRegistryEntry> type_registry = []() {
	std::unordered_map<size_t, TypeRegistryEntry> ret;
	TypeRegistryEntry e;
	e.elemCount = 1;
	e.printFunc = fillTableWithData<int8_t>;
	for (auto& it : { &typeid(i8x1),&typeid(i8x2), &typeid(i8x4), &typeid(i8x8), &typeid(i8x16), &typeid(i8x32), &typeid(i8x64) })
	{
		ret[it->hash_code()] = e;
		e.elemCount *= 2;
	}
	e.elemCount = 1;
	e.printFunc = fillTableWithData<int16_t>;
	for (auto& it : { &typeid(i16x1),&typeid(i16x2), &typeid(i16x4), &typeid(i16x8), &typeid(i16x16), &typeid(i16x32), &typeid(i16x64) })
	{
		ret[it->hash_code()] = e;
		e.elemCount *= 2;
	}
	e.elemCount = 1;
	e.printFunc = fillTableWithData<int32_t>;
	for (auto& it : { &typeid(i32x1),&typeid(i32x2), &typeid(i32x4), &typeid(i32x8), &typeid(i32x16), &typeid(i32x32), &typeid(i32x64) })
	{
		ret[it->hash_code()] = e;
		e.elemCount *= 2;
	}
	e.elemCount = 1;
	e.printFunc = fillTableWithData<int64_t>;
	for (auto& it : { &typeid(i64x1),&typeid(i64x2), &typeid(i64x4), &typeid(i64x8), &typeid(i64x16), &typeid(i64x32), &typeid(i64x64) })
	{
		ret[it->hash_code()] = e;
		e.elemCount *= 2;
	}

	

	e.elemCount = 1;
	e.printFunc = fillTableWithData<uint8_t>;
	for (auto& it : { &typeid(u8x1),&typeid(u8x2), &typeid(u8x4), &typeid(u8x8), &typeid(u8x16), &typeid(u8x32), &typeid(u8x64) })
	{
		ret[it->hash_code()] = e;
		e.elemCount *= 2;
	}
	e.elemCount = 1;
	e.printFunc = fillTableWithData<uint16_t>;
	for (auto& it : { &typeid(u16x1),&typeid(u16x2), &typeid(u16x4), &typeid(u16x8), &typeid(u16x16), &typeid(u16x32), &typeid(u16x64) })
	{
		ret[it->hash_code()] = e;
		e.elemCount *= 2;
	}
	e.elemCount = 1;
	e.printFunc = fillTableWithData<uint32_t>;
	for (auto& it : { &typeid(u32x1),&typeid(u32x2), &typeid(u32x4), &typeid(u32x8), &typeid(u32x16), &typeid(u32x32), &typeid(u32x64) })
	{
		ret[it->hash_code()] = e;
		e.elemCount *= 2;
	}
	e.elemCount = 1;
	e.printFunc = fillTableWithData<uint64_t>;
	for (auto& it : { &typeid(u64x1),&typeid(u64x2), &typeid(u64x4), &typeid(u64x8), &typeid(u64x16), &typeid(u64x32), &typeid(u64x64) })
	{
		ret[it->hash_code()] = e;
		e.elemCount *= 2;
	}



	e.elemCount = 1;
	e.printFunc = fillTableWithData<float>;
	for (auto& it : { &typeid(f32x1),&typeid(f32x2), &typeid(f32x4), &typeid(f32x8), &typeid(f32x16), &typeid(f32x32), &typeid(f32x64) })
	{
		ret[it->hash_code()] = e;
		e.elemCount *= 2;
	}
	e.elemCount = 1;
	e.printFunc = fillTableWithData<double>;
	for (auto& it : { &typeid(f64x1),&typeid(f64x2), &typeid(f64x4), &typeid(f64x8), &typeid(f64x16), &typeid(f64x32), &typeid(f64x64) })
	{
		ret[it->hash_code()] = e;
		e.elemCount *= 2;
	}
	e.elemCount = 1;
	e.printFunc = fillTableWithData<fp16_t>;
	for (auto& it : { &typeid(fp16x1),&typeid(fp16x2), &typeid(fp16x4), &typeid(fp16x8), &typeid(fp16x16), &typeid(fp16x32), &typeid(fp16x64) })
	{
		ret[it->hash_code()] = e;
		e.elemCount *= 2;
	}
	return ret;
	}();

void Reporter::reportFailure(const ReporterInput& input)
{
	std::ostringstream ss;
	std::string testName = (!input.testName.empty() ? input.testName : "(unnamed test)");
	ss << "\nTEST FAILED: " << testName << "\n";
	ss << "Source location: " << input.loc.function_name() << "\n";

	for (size_t i = 0; i < input.args.size(); ++i)
	{
		ss << "Input argument " << i << " type: " << input.args[i].type->name() << "\n";
	}
	ss << "Candidate return type: " << input.candRet.type->name() << "\n";
	ss << "Reference return type: " << input.refRet.type->name() << "\n";
	if (input.candRet.type != input.refRet.type) ss << "THE RETURN TYPES FOR CANDIDATE AND REFERENCE ARE NOT EQUAL!!!\n";

	// and calculate max width for each future column.
	Table table;
	table.cellAt(0, 0).value = "Index";

	//Stringify all inputs to hex and decimal formats
	//Columns: 0: index, [1..argCount]: arg hex, argCount+1: cand hex, argCount+2: ref hex, argCount+3: pipes, [argCount+4...2*argCount+3]: arg dec, 2*argCount+4: cand dec, 2*argCount+5 ref dec
	//TODO: check correctness of columns and make alias system
	size_t argCount = input.args.size();
	for (size_t argInd = 0; argInd < argCount; ++argInd)
	{
		//Headers
		size_t hexCol = argInd + 1;
		size_t decCol = argCount + 4 + argInd;
		table.cellAt(0, hexCol).value = "A" + std::to_string(argInd) + " hex";
		table.cellAt(0, decCol).value = "A" + std::to_string(argInd) + " dec";

		auto& arg = input.args[argInd];
		type_registry.at(arg.type->hash_code()).printFunc(arg.bytes.data(), arg.bytes.size(), table, hexCol, decCol);
	}
	size_t rows, cols;
	table.getSize(&rows, &cols);
	table.cellAt(0, argCount + 3).value = "|";
	for (size_t i = 0; i < rows - 1; ++i)
	{
		table.cellAt(i + 1, 0).value = std::to_string(i);
		table.cellAt(i + 1, argCount + 3).value = "|";
	}

	table.cellAt(0, argCount + 1).value = "Cand hex";
	table.cellAt(0, argCount + 2).value = "Ref hex";
	table.cellAt(0, 2 * argCount + 4).value = "Cand dec";
	table.cellAt(0, 2 * argCount + 5).value = "Ref dec";

	type_registry.at(input.candRet.type->hash_code()).printFunc(input.candRet.bytes.data(), input.candRet.bytes.size(), table, argCount + 1, 2 * argCount + 4);
	type_registry.at(input.refRet.type->hash_code()).printFunc(input.refRet.bytes.data(), input.refRet.bytes.size(), table, argCount + 2, 2 * argCount + 5);
	ss << table.toString() << "\n";

	//Final output to console and files
	std::cout << ss.str();

	std::string cleanedTestName = testName;
	for (auto& it : "<>,.:/\\*?|\"")
	{
		for (auto& c : cleanedTestName) if (c == it) c = '_';
	}
	auto it = this->logFiles.find(cleanedTestName);
	if (it == this->logFiles.end()) this->logFiles[cleanedTestName] = std::ofstream(std::string("test_logs/test_") + cleanedTestName + ".log");
	this->logFiles[cleanedTestName] << ss.str() << "\n\n";
}
