#include "Table.h"
#include <sstream>

TableCell& Table::cellAt(size_t row, size_t col)
{
	return cells[std::make_pair(row, col)];
}

void Table::getSize(size_t* outRows, size_t* outCols) const
{
	size_t rowCount = 0, colCount = 0;
	for (auto& it : cells)
	{
		rowCount = std::max(rowCount, it.first.first + 1);
		colCount = std::max(colCount, it.first.second + 1);
	}
	if (outRows) *outRows = rowCount;
	if (outCols) *outCols = colCount;
}

std::string Table::toString() const
{
	std::ostringstream oss;
	size_t rowCount, colCount;
	this->getSize(&rowCount, &colCount);

	std::vector<size_t> colWidths(colCount);
	for (size_t colI = 0; colI < colCount; ++colI)
	{
		for (size_t rowI = 0; rowI < rowCount; ++rowI)
		{
			auto it = this->cells.find(std::make_pair(rowI, colI));
			if (it == this->cells.end()) continue;

			const TableCell& cell = it->second;
			colWidths[colI] = std::max(colWidths[colI], cell.value.length());
		}
	}


	for (size_t rowI = 0; rowI < rowCount; ++rowI)
	{
		for (size_t colI = 0; colI < colCount; ++colI)
		{
			size_t colW = colWidths[colI] + 2;
			auto it = this->cells.find(std::make_pair(rowI, colI));
			if (it == this->cells.end())
			{
				oss << std::string(colW, ' ');
				continue;
			}

			const TableCell& cell = it->second;
			size_t valueLen = cell.value.length();
			oss << std::string(colW - valueLen, ' ') << cell.value;
		}
		oss << "\n";
	}
	return oss.str();
}