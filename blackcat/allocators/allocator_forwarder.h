/*
 * Allocator_Wrapper.h
 *
 *  Created on: Jan 20, 2020
 *      Author: joseph
 */

#ifndef BLACKCAT_ALLOCATORS_ALLOCATOR_FORWADER_H_
#define BLACKCAT_ALLOCATORS_ALLOCATOR_FORWADER_H_

#include "allocator_traits.h"

namespace bc {
namespace allocators {

template<class Allocator>
struct Allocator_Forwarder: Allocator {

	using traits = allocator_traits<Allocator>;
	using value_type = typename traits::value_type;
	using system_tag = typename traits::system_tag;
	using pointer = typename traits::pointer;
	using const_pointer = typename traits::const_pointer;
	using void_pointer = typename traits::void_pointer;
	using difference_type = typename traits::difference_type;
	using size_type = typename traits::size_type;
	using is_always_equal = typename traits::is_always_equal;

	using Allocator::operator==;
	using Allocator::operator!=;

	using propagate_on_container_copy_assignment =
			typename traits::propagate_on_container_copy_assignment;
	using propagate_on_container_move_assignment =
			typename traits::propagate_on_container_move_assignment;
	using propagate_on_container_swap =
			typename traits::propagate_on_container_swap;

	template<class AltValueType>
	struct rebind {
		using other = Allocator_Forwarder<
				typename traits::template rebind_alloc<AltValueType>>;
	};

	auto select_on_container_copy_construction() {
		 return traits::select_on_container_copy_construction(*this);
	}

	pointer allocate(size_type size) {
		return Allocator::allocate(size);
	}

	void deallocate(pointer ptr, size_type size) {
		Allocator::deallocate(ptr, size);
	}

	template<class... Args >
	void construct(pointer ptr, Args&&... args ) {
		traits::construct(*this, ptr, std::forward<Args>(args)...);
	}

	void destroy(pointer ptr) {
		traits::destroy(*this, ptr);
	}
};

}
}

#endif /* ALLOCATOR_WRAPPER_H_ */
