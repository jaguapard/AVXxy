#include <iostream>
#include "include/avxxy.h"

using namespace AVXXY_NAMESPACE;
using namespace internals;
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

	//This is not how users will use it, but they can if they want to be really specific. This will be dressed in much nicer calls, Dispatcher and tags should be basically invisible to users
	f32x16 read_x2 = Dispatcher<FS_current>::run(op_add{}, read, read);
	std::cout << "Read + read as f32x16: " << read_x2 << "\n";

	f32x16 read_x3_add_func = add(read, read_x2);
	std::cout << "Read * 3 through add func: " << read_x3_add_func << "\n";

	i32x16 cvt = Dispatcher<FS_current>::run(op_cvt<int>{}, read_x2);
	std::cout << "Read * 2 as ints: " << cvt << "\n";

	f32x64 ff;
	for (size_t i = 0; i < 64; ++i) ff[i] = i;
	std::cout << "Mask mov: " << mask_mov(f32x64(56), 0x5555555555555555, ff) << "\n";

	system("pause");
	return 0;
}