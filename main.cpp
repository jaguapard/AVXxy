
#include "avxxy/avxxy.h"
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

	auto avx512_add = backends::AVX512::add(read, read);
	auto avx2_add = backends::AVX2::add(read, read);
	auto sse41_add = backends::SSE41::add(read, read);
	auto sse2_add = backends::SSE2::add(read, read);
	auto scalar_add = backends::Scalar::add(read, read);

	std::cout << "AVX512: " << avx512_add << "\n";
	std::cout << "AVX2: " << avx2_add << "\n";
	std::cout << "SSE4.1: " << sse41_add << "\n";
	std::cout << "SSE2: " << sse2_add << "\n";
	std::cout << "Scalar: " << scalar_add << "\n";

	auto avx512_cvtps_pd = backends::AVX512::cvt<double>(read);
	std::cout << "AVX512 ps to pd: " << avx512_cvtps_pd << "\n";

	system("pause");
	return 0;
}