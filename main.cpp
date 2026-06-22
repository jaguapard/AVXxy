#include <iostream>
#include "include/avxxy.h"

using namespace AVXXY_NAMESPACE;
using namespace internals;
int main()
{
	std::cout << "Compile target feature set: \n" << FS_compile_target << "\n\n";
	std::cout << "Current feature set: " << FS_current << "\n\n";
	char buf[64] = { 0 };
	std::cout << "Input anything:\n";
	fgets(buf, sizeof(buf), stdin);

	f32x64 ff;
	f32x16 read;
	size_t zcnt = 0;
	for (auto c : buf) if (!c) ++zcnt;
	if (zcnt < sizeof(buf) - 1)
	{
		memcpy(&read, buf, std::min(sizeof(read), sizeof(buf)));
		for (size_t i = 0; i < 4; ++i)
		{
			float* pp = &ff[0];
			memcpy(pp + i * 16, buf, 64);
		}
	}
	else
	{
		for (size_t i = 0; i < 64; ++i) ff[i] = i;
		read = f32x16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
	}

	std::cout << "Read as f32x16: " << read << "\n";

	u8x64 smoll;
	memcpy(&smoll, buf, 64);
	u8x64 smoll_x2 = ISA_Scalar::eval<op_gather<uint8_t, 64>>(&read, smoll, mask_t<uint8_t,64>(0x32451234), smoll);
	std::cout << smoll_x2;
	//f32x16 read_x2 = add(read, read);
	//std::cout << "Read + read as f32x16: " << read_x2 << "\n";

	//f64x16 bb = vcvt<double>(read);
	//std::cout << bb << "\n";
#if 0
	

	

	//This is not how users will use it, but they can if they want to be really specific. This will be dressed in much nicer calls, Dispatcher and tags should be basically invisible to users
	f32x16 read_x2 = Dispatcher::run(op_add{}, read, read);
	std::cout << "Read + read as f32x16: " << read_x2 << "\n";

	f32x16 read_x3_add_func = add(read, read_x2);
	std::cout << "Read * 3 through add func: " << read_x3_add_func << "\n";

	i32x16 cvt = Dispatcher::run(op_cvt<int>{}, read_x2);
	std::cout << "Read * 2 as ints: " << cvt << "\n";

	auto div = read / read;
	
	auto mask_m = mask_mov(f32x64(56), 0x5555555555555555, ff);
	std::cout << "Mask mov: " << mask_m << "\n";

	f32x8 readLo = vcast<f32x8>(read);
	f32x8 readLo_x2 = readLo + readLo;
	std::cout << "Read low half x2:" << readLo_x2 << "\n";
#endif
	system("pause");
	return 0;
}