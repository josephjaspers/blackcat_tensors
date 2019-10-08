/*
 * Null_Allocator.h
 *
 *  Created on: Oct 4, 2019
 *      Author: joseph
 */

#ifndef BLACKCATTENSORS_ALLOCATORS_NULL_ALLOCATOR_H_
#define BLACKCATTENSORS_ALLOCATORS_NULL_ALLOCATOR_H_

namespace BC {
namespace allocators {

template<class SystemTag, class ValueType>
struct Null_Allocator {

	using system_tag =  SystemTag;
	using value_type = ValueType;
	using size_t = BC::size_t;

	value_type* allocate(size_t sz) {
		return nullptr;
	}

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



#endif /* NULL_ALLOCATOR_H_ */
