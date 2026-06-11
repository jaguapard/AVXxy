#include <iostream>
#include "include/avxxy.h"

using namespace AVXXY_NAMESPACE;

int main()
{
	char buf[64] = { 0 };
	std::cout << "Input anything:\n";
	fgets(buf, sizeof(buf), stdin);

	f32x16 read;
	size_t zcnt = 0;
	for (auto c : buf) if (!c) ++zcnt;
	if (zcnt < sizeof(buf) - 1) memcpy(&read, buf, std::min(sizeof(read), sizeof(buf)));
	else read = f32x16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

	std::cout << "Read as f32x16: " << read << "\n";


	system("pause");
	return 0;
}