/*
 * Shared_Allocator.h
 *
 *  Created on: Sep 22, 2019
 *      Author: joseph
 */

#ifndef BLACKCATTENSORS_ALLOCATORS_FANCY_SHARED_ALLOCATOR_H_
#define BLACKCATTENSORS_ALLOCATORS_FANCY_SHARED_ALLOCATOR_H_

namespace BC {
namespace allocators {

/**
 * An allocator that wraps another with a shared ptr and forwards functions to it.
 */
template<class Allocator>
struct Shared_Allocator {

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
			Shared_Allocator<
			typename BC::allocator_traits<Allocator>::template rebind_alloc<altT>>;
	};


private:
	std::shared_ptr<Allocator> m_allocator;
public:


	template<class U>
	Shared_Allocator(const Shared_Allocator<U>& alloc):
		m_allocator(
				typename BC::allocator_traits<Allocator>::
					template rebind_alloc<value_type>(*(alloc.m_allocator))) {}

	Shared_Allocator(Allocator allocator):
		m_allocator(allocator) {}

	Shared_Allocator():
	m_allocator(new Allocator()) {}


//	template<class... Args>
//	Shared_Allocator(Args&&... args):
//	m_allocator(new Allocator(args...)) {}

	template<class... Args>
	Shared_Allocator(std::shared_ptr<Allocator> allocator):
		m_allocator(allocator) {}


	Shared_Allocator(const Shared_Allocator&) = default;
	Shared_Allocator(Shared_Allocator&&) = default;

	Shared_Allocator& operator = (const Shared_Allocator&) = default;
	Shared_Allocator& operator = (Shared_Allocator&&) = default;

	value_type* allocate(int size) {
		return m_allocator->allocate(size);
    }

    void deallocate(value_type* t, BC::size_t  size) {
		m_allocator->deallocate(t, size);
     }

    template<class U>
    bool operator == (const Shared_Allocator<U>& other) const {
    	return *m_allocator == *(other.m_allocator);
    }

    template<class U>
    bool operator != (const Shared_Allocator<U>& other) const {
    	return *m_allocator != *(other.m_allocator);
    }
};

}
}


#endif /* SHARED_ALLOCATOR_H_ */
