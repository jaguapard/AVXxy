#include <iostream>
#include "include/avxxy.h"

using namespace avxxy;
using namespace internals;
//TODO: verify that it works
void scatterToFrameBuffer(const std::array<f32x16, 4>& colors, i32x16 x, i32x16 y, mask16d mask, void* frameBuf, int framebufW)
{
	i32x16 scatterInd = y * framebufW + x;
	fp16x16 fp16_r = colors[0];
	fp16x16 fp16_g = colors[1];
	fp16x16 fp16_b = colors[2];
	fp16x16 fp16_a = colors[3]; //TODO: can be forced to 1 and moved later

	fp16x32 fp16_rg = { fp16_r, fp16_g };
	fp16x32 fp16_ba = { fp16_b, fp16_a };

	u64x8 rgba0_7 = vcast<u64x8>(permx2(fp16_rg, fp16_ba, u16x32(0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51, 4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55)));
	u64x8 rgba8_15 = vcast<u64x8>(permx2(fp16_rg, fp16_ba, u16x32(8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59, 12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63)));
	//ISA_Scalar::eval<op_scatter<8>>(rgba0_7, frameBuf, scatterInd.lo(), mask.lo());
	scatter(rgba0_7, frameBuf, scatterInd.lo(), mask.lo());
	//scatter(rgba8_15, frameBuf, scatterInd.hi(), mask.hi());
}


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

	f32x16 rr = abs(read);
	int xx = 0;
	std::cout << rr;

	mask32d x = 0x434234;
	auto z123187 = vpopcnt(read);
	std::cout << "\n" << z123187;
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