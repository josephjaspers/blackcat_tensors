/*
 * Null_Allocator.h
 *
 *  Created on: Oct 4, 2019
 *      Author: joseph
 */

#ifndef BLACKCATTENSORS_ALLOCATORS_NULL_ALLOCATOR_H_
#define BLACKCATTENSORS_ALLOCATORS_NULL_ALLOCATOR_H_

namespace bc {
namespace allocators {

template<class SystemTag, class ValueType>
struct Null_Allocator {

	using system_tag =  SystemTag;
	using value_type = ValueType;
	using size_t = bc::size_t;

	Null_Allocator()=default;
	Null_Allocator(const Null_Allocator&)=default;
	Null_Allocator(Null_Allocator&&)=default;

	template<class AltT>
	Null_Allocator(Null_Allocator<SystemTag, AltT> copy) {}
	value_type* allocate(size_t sz) { return nullptr; }

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

#endif /* NULL_ALLOCATOR_H_ */
