/*
 * Atomic_Allocator.h
 *
 *  Created on: Sep 21, 2019
 *      Author: joseph
 */

#ifndef BLACKCATTENSORS_ALLOCATORS_FANCY_ATOMIC_ALLOCATOR_H_
#define BLACKCATTENSORS_ALLOCATORS_FANCY_ATOMIC_ALLOCATOR_H_

#include <mutex>

namespace BC {
namespace allocators {

/**
 * An allocator that wraps another but makes accesses to its functions atomic.
 */
template<class Allocator>
struct Atomic_Allocator {

    using system_tag = typename BC::allocator_traits<Allocator>::system_tag;	//BC tag

    using value_type = typename Allocator::value_type;
    using pointer = value_type*;
    using const_pointer = value_type*;
    using size_type = int;
    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::false_type;
    using propagate_on_container_swap = std::false_type;
    using is_always_equal = std::true_type;

	template<class altT>
	struct rebind { using other =
			Atomic_Allocator<
			typename BC::allocator_traits<Allocator>::template rebind_alloc<altT>>;
	};

	template<class U>
	Atomic_Allocator(const Atomic_Allocator<U>& alloc):
		m_allocator(alloc.m_allocator) {}

	Atomic_Allocator(Allocator allocator):
		m_allocator(allocator) {}

	Atomic_Allocator() = default;
	Atomic_Allocator(const Atomic_Allocator&) = default;
	Atomic_Allocator(Atomic_Allocator&&) = default;

	Atomic_Allocator& operator = (const Atomic_Allocator&) = default;
	Atomic_Allocator& operator = (Atomic_Allocator&&) = default;

	Allocator m_allocator;
	std::mutex m_locker;

	value_type* allocate(int size) {
		m_locker.lock();
		value_type* ptr = m_allocator.allocate(size);
		m_locker.unlock();
		return ptr;
    }

    void deallocate(value_type* t, BC::size_t  size) {
		m_locker.lock();
		m_allocator.deallocate(t, size);
		m_locker.unlock();
    }

    template<class U>
    bool operator == (const Atomic_Allocator<U>& other) const {
    	return m_allocator == other.m_allocator;
    }

    template<class U>
    bool operator != (const Atomic_Allocator<U>& other) const {
    	return m_allocator != other.m_allocator;
    }
};

}
}





#endif /* ATOMIC_ALLOCATOR_H_ */
