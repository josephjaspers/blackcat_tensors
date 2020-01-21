/*
 * Logging_Allocator.h
 *
 *  Created on: Jan 20, 2020
 *      Author: joseph
 */

#ifndef BLACKCAT_TENSORS_ALLOCATORS_LOGGING_ALLOCATOR_H_
#define BLACKCAT_TENSORS_ALLOCATORS_LOGGING_ALLOCATOR_H_

#include "allocator_forwarder.h"

namespace bc {
namespace allocators {

namespace detail {
	struct log_info {
		unsigned max_allocated; //maximum number of bytes allocated
		unsigned current_allocated;
	};
}

template<class Allocator>
struct Logging_Allocator: Allocator_Forwarder<Allocator> {

	using parent_type = Allocator_Forwarder<Allocator>;
	using value_type = typename parent_type::value_type;
	using pointer = typename parent_type::pointer;

	template<class altT>
	struct rebind {
		using other = Logging_Allocator<
				typename parent_type::template rebind<altT>::other>;
	};

	std::shared_ptr<detail::log_info> info =
			std::shared_ptr<detail::log_info>(new detail::log_info {0, 0});

	template<class U>
	Logging_Allocator(const Logging_Allocator<U>& other): info(other.info) {}
	Logging_Allocator() = default;
	Logging_Allocator(const Logging_Allocator&) = default;
	Logging_Allocator(Logging_Allocator&&) = default;

	Logging_Allocator& operator = (const Logging_Allocator&) = default;
	Logging_Allocator& operator = (Logging_Allocator&&) = default;

	pointer allocate(int size) {
		info->current_allocated += size * sizeof(value_type);
		if (info->current_allocated > info->max_allocated){
			info->max_allocated = info->current_allocated;
		}
		return parent_type::allocate(size);
	}

	void deallocate(pointer ptr, bc::size_t size) {
		BC_ASSERT(info->current_allocated >= size * sizeof(value_type),
				"BC_DEALLOCATION ERROR, DOUBLE DUPLICATION")
		info->current_allocated -= size * sizeof(value_type);

		parent_type::deallocate(ptr, size);
	}

};

}
}




#endif /* LOGGING_ALLOCATOR_H_ */
