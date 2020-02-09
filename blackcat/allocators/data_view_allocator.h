/*
 * data_view_allocator.h
 *
 *  Created on: Feb 8, 2020
 *      Author: joseph
 */

#ifndef BLACKCAT_ALLOCATORS_DATA_VIEW_ALLOCATOR_H_
#define BLACKCAT_ALLOCATORS_DATA_VIEW_ALLOCATOR_H_

#include "basic_allocators.h"

namespace bc {
namespace allocators {

template<
	class SystemTag,
	class ValueType,
	class AltAllocator=bc::allocators::Allocator<SystemTag, ValueType>>
struct Data_View_Allocator
{
	using system_tag = SystemTag;
	using value_type = ValueType;
	using size_t = bc::size_t;

	AltAllocator allocator;
	ValueType* data_ptr;
	bool is_allocated = false;
	int size;
	Data_View_Allocator(const Data_View_Allocator& other):
		allocator(other.allocator),
		data_ptr(other.data_ptr),
		is_allocated(true),
		size(0) {}

	Data_View_Allocator(value_type* data_ptr, std::size_t data_sz,
			AltAllocator fallback_allocator=AltAllocator()):
		data_ptr(data_ptr),
		size(data_ptr) {}

	Data_View_Allocator(const Data_View_Allocator& other):
		allocator(other.allocator),
		data_ptr(other.data_ptr),
		is_allocated(true),
		size(0) {}

	Data_View_Allocator(Data_View_Allocator&& other):
		allocator(std::move(other.allocator)),
		data_ptr(other.data_ptr),
		is_allocated(other.is_allocated),
		size(other.size)
	{
		other.is_allocated = true;
	}

	Data_View_Allocator& operator =(const Data_View_Allocator& other){
		allocator = other.allocator;
		return *this;
	}

	Data_View_Allocator& operator =(Data_View_Allocator&& other){
		allocator = std::move(other.allocator);
		return *this;
	}


	value_type* allocate(size_t sz)
	{
		if (!is_allocated && size <= sz) {
			return data_ptr;
		} else {
			return allocator.allocate(sz);
		}
	}

	template<class AltT>
	struct rebind {
		using other = Null_Allocator<system_tag, AltT>;
	};

	void deallocate(value_type* ptr, size_t sz) {
		BC_ASSERT(ptr==nullptr, "Null_Allocator passed a non-null ptr");
	}

	template<class U>
	constexpr bool operator ==(const Null_Allocator<SystemTag, U>&) const {
		return true;
	}

	template<class U>
	constexpr bool operator !=(const Null_Allocator<SystemTag, U>&) const {
		return false;
	}
};

}
}

#endif /* DATA_VIEW_ALLOCATOR_H_ */
