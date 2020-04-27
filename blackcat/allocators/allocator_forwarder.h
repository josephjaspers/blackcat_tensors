/*
 * Allocator_Wrapper.h
 *
 *  Created on: Jan 20, 2020
 *      Author: joseph
 */

#ifndef BLACKCAT_ALLOCATORS_ALLOCATOR_FORWADER_H_
#define BLACKCAT_ALLOCATORS_ALLOCATOR_FORWADER_H_

#include "allocator_traits.h"

#ifdef __CUDACC__
#include <thrust/device_ptr.h>
#include <thrust/device_allocator.h>
#endif

namespace bc {
namespace allocators {

template<class Allocator>
struct Allocator_Forwarder {

	using traits     = allocator_traits<Allocator>;
	using value_type = typename traits::value_type;
	using system_tag = typename traits::system_tag;
	using pointer    = typename traits::pointer;
	using const_pointer   = typename traits::const_pointer;
	using void_pointer    = typename traits::void_pointer;
	using reference  = value_type&;
	using const_reference = const value_type&;
	using difference_type = typename traits::difference_type;
	using size_type       = typename traits::size_type;
	using is_always_equal = typename traits::is_always_equal;

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

private:
	Allocator m_allocator;
public:

	template<class... Args>
	Allocator_Forwarder(Args&&... args): m_allocator(args...) {}

	template<class AltAllocator>
	Allocator_Forwarder(const Allocator_Forwarder<AltAllocator>& other):
		m_allocator(other.m_allocator) {}


	auto select_on_container_copy_construction() {
		return traits::select_on_container_copy_construction(m_allocator);
	}

	pointer allocate(size_type size) {
		return m_allocator.allocate(size);
	}

	void deallocate(pointer ptr, size_type size) {
		m_allocator.deallocate(ptr, size);
	}

	template<class... Args >
	void construct(pointer ptr, Args&&... args ) {
		traits::construct(m_allocator, ptr, std::forward<Args>(args)...);
	}

	void destroy(pointer ptr) {
		traits::destroy(m_allocator, ptr);
	}

	template<class AltAllocator>
	bool operator == (const AltAllocator& other) {
		return m_allocator == other;
	}

	template<class AltAllocator>
	bool operator != (const AltAllocator& other) {
		return m_allocator != other;
	}
};

#ifdef __CUDACC__

template<class ValueType, class Allocator>
struct Thrust_Allocator_Forwarder {

	using traits     = thrust::device_allocator<ValueType>;
	using value_type = typename traits::value_type;
	using system_tag = bc::device_tag;
	using pointer    = typename traits::pointer;
	using const_pointer   = const pointer;
	using void_pointer    = typename traits::void_pointer;
	using reference       = typename traits::reference;
	using const_reference = typename traits::const_reference;
	using difference_type = typename traits::difference_type;
	using size_type       = typename traits::size_type;
	using is_always_equal = typename bc::allocator_traits<Allocator>::is_always_equal;

	using propagate_on_container_copy_assignment =
			typename traits::propagate_on_container_copy_assignment;

	using propagate_on_container_move_assignment =
			typename traits::propagate_on_container_move_assignment;

	using propagate_on_container_swap =
			typename traits::propagate_on_container_swap;

	template<class AltValueType>
	struct rebind {
		using other = Thrust_Allocator_Forwarder<AltValueType,
			typename traits::template rebind_alloc<AltValueType>>;
	};

private:
	Allocator m_allocator;
public:

	Thrust_Allocator_Forwarder() {}

	template<class... Args>
	Thrust_Allocator_Forwarder(const Args&... args):
		m_allocator(args...) {}

	Thrust_Allocator_Forwarder(const Thrust_Allocator_Forwarder& other):
		m_allocator(other.m_allocator) {}

	Thrust_Allocator_Forwarder(Thrust_Allocator_Forwarder&& other):
		m_allocator(std::move(other.m_allocator)) {}

	Thrust_Allocator_Forwarder& operator = (const Thrust_Allocator_Forwarder& other) {
		m_allocator = other.m_allocator;
		return *this;
	}

	Thrust_Allocator_Forwarder& operator = (Thrust_Allocator_Forwarder&& other) {
		m_allocator = std::move(other.m_allocator);
		return *this;
	}

	template<class AltVt, class AltAllocator>
	Thrust_Allocator_Forwarder(const Thrust_Allocator_Forwarder<AltVt, AltAllocator>& other):
		m_allocator(other.m_allocator) {}


	auto select_on_container_copy_construction() {
		return traits::select_on_container_copy_construction(m_allocator);
	}

	pointer allocate(size_type size) {
		return pointer(m_allocator.allocate(size));
	}

	void deallocate(pointer ptr, size_type size) {
		value_type* vt_ptr = ptr.get();
		m_allocator.deallocate(vt_ptr, size);
	}

	template<class... Args >
	void construct(pointer ptr, Args&&... args ) {
		traits::construct(m_allocator, ptr.get(), std::forward<Args>(args)...);
	}

	void destroy(pointer ptr) {
		traits::destroy(m_allocator, ptr.get());
	}

	template<class AltAllocator>
	bool operator == (const AltAllocator& other) {
		return m_allocator == other;
	}

	template<class AltAllocator>
	bool operator != (const AltAllocator& other) {
		return m_allocator != other;
	}
};

template<class Allocator>
struct allocator_to_thrust_allocator {
	using value_type = typename bc::allocator_traits<Allocator>::value_type;
	using type = Thrust_Allocator_Forwarder<value_type, Allocator>;
};

template<class Vt, class Allocator>
struct allocator_to_thrust_allocator<Thrust_Allocator_Forwarder<Vt, Allocator>> {
	using value_type = Vt;
	using type = Thrust_Allocator_Forwarder<value_type, Allocator>;
};

template<class Allocator>
using allocator_to_thrust_allocator_t = typename allocator_to_thrust_allocator<Allocator>::type;



//template<class Vt, class Allocator>
//struct Thrust_Allocator_Forwarder<Vt, Thrust_Allocator_Forwarder<Vt, Allocator>> {
//	Thrust_Allocator_Forwarder() {
//		static_assert(false, "Invalid type");
//	}
//};

#endif

}
}

#endif /* ALLOCATOR_WRAPPER_H_ */
