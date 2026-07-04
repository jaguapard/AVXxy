#pragma once
#include "../include/namespace.h"
#include <stdexcept>
#include <span>
namespace AVVXY_NAMESPACE
{
	namespace internals
	{
		//Represents an address in memory. Convertible to and constructible from any pointer type
		struct TypelessPtr
		{
		private:
			size_t addr;
		public:
			TypelessPtr() {};
			TypelessPtr(void* p)
			{
				this->addr = size_t(p);
			}
			template<typename P>
			operator P* () const
			{
				return (P*)(this->addr);
			}

			TypelessPtr operator+(size_t n) const
			{
				TypelessPtr ret;
				ret.addr = addr + n;
				return ret;
			}

			//Returns this pointer aligned to boundary. The result's address is not greater than this pointer's one.
			//Does not change address if pointer is already aligned
			TypelessPtr alignDec(size_t alignmentInBytes) const
			{
				if (addr % alignmentInBytes == 0) return *this;
				TypelessPtr ret;
				ret.addr = this->addr - this->addr % alignmentInBytes;
				return ret;
			}
			//Returns this pointer aligned to boundary. The result's address is not less than this pointer's one.
			//Does not change address if pointer is already aligned
			TypelessPtr alignInc(size_t alignmentInBytes) const
			{
				if (addr % alignmentInBytes == 0) return *this;
				TypelessPtr ret = alignDec(alignmentInBytes);
				ret.addr += alignmentInBytes;
				return ret;
			}

			//Returns pointer aligned to alignmentRequirementInBytes boundary.
			//If resultant memory address is less than the beginning of the span, advances it by one alignment requirement forward
			//If after that the pointer is out of span's bounds, throws an exception. This can only happen if span content size is smaller than alignment requirement
			template<typename T>
			TypelessPtr alignBounded(size_t alignmentInBytes, const std::span<T>& bounds) const
			{
				size_t boundLo = size_t(bounds.data());
				size_t boundHi = size_t(&bounds.back()) + sizeof(T);

				TypelessPtr ret = this->alignDec(alignmentInBytes);
				if (ret.addr < boundLo) ret.addr += alignmentInBytes;
				if (ret.addr >= boundHi) throw std::runtime_error("Can't align pointer to bounds: too small bounds!");
				return ret;
			}
		};
	}
}