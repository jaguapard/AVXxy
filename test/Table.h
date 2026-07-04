#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <map>

struct ExtraData
{
	std::vector<std::byte> data;
	const type_info* type;
	ExtraData() = default;

	template<typename T>
	ExtraData(const T& inp)
	{
		static_assert(std::is_trivially_copyable_v<T>);
		type = &typeid(T);
		data.resize(sizeof(T));
		memcpy(data.data(), &inp, sizeof(T));
	}
};
struct TableCell
{
	std::string value;
	ExtraData extraData;
};
struct Table
{
	//Returns a reference to cell inside this table. It is recommended to not store it anywhere, since adding new cells can make it dangling
	//If cell doesn't exist at that location, it will be created and default-initialized before returning
	TableCell& cellAt(size_t row, size_t col);

	//Outputs number of rows and columns. If and output parameter is set to null, that value will be discarded
	//Returns 0 for both if this table is empty.
	//The number of cols and rows is equal to maximum index of row and col + 1
	//This means that it is not necessary to create all rows and columns manually, you can create a sparse table,
	//and it will adjust itself.
	void getSize(size_t* outRows, size_t* outCols) const;

	//Aligns all columns by their maximum element size and outputs them all as a single string
	std::string toString() const;
private:
	std::map<std::pair<size_t, size_t>, TableCell> cells;
};
