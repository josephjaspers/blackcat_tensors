/*
 * Workspace.h
 *
 *  Created on: Mar 16, 2019
 *      Author: joseph
 */

#ifndef BC_CORE_CONTEXT_WORKSPACE_H_
#define BC_CORE_CONTEXT_WORKSPACE_H_

#include "Polymorphic_Allocator.h"
#include <vector>

namespace BC {
namespace allocators {
namespace fancy {
namespace detail {

template<class SystemTag>
class Workspace_Base {

	using system_tag = SystemTag;

	std::size_t m_memptr_sz=0;
	std::size_t m_curr_index=0;

	Byte* m_memptr=nullptr;

	static Polymorphic_Allocator<SystemTag, BC::allocators::Byte>& get_default_allocator() {
		static Polymorphic_Allocator<SystemTag, BC::allocators::Byte> default_allocator;
		return default_allocator;
	}

	Polymorphic_Allocator<SystemTag, Byte> m_allocator = get_default_allocator();

public:
	Workspace_Base(std::size_t sz=0) : m_memptr_sz(sz){
		if (sz)
			m_memptr = m_allocator.allocate(sz);
	}
	void reserve(std::size_t sz)  {
		BC_ASSERT(m_curr_index==0,
				"Workspace reserve called while memory is still allocated");

		if (m_memptr_sz < sz) {
			m_allocator.deallocate(m_memptr, m_memptr_sz);
			m_memptr = m_allocator.allocate(sz);
			m_memptr_sz = sz;
		}
	}
	void free() {
		BC_ASSERT(m_curr_index==0,
				"Workspace free called while memory is still allocated");
		m_allocator.deallocate(m_memptr, m_memptr_sz);
		m_memptr_sz = 0;
		m_memptr = nullptr;
	}

	size_t available_bytes() const {
		 return m_memptr_sz - m_curr_index;
	}
	size_t allocated_bytes() const {
		 return m_curr_index;
	}
	size_t reserved_bytes() const {
		 return m_memptr_sz;
	}

	template<class Allocator>
	void set_allocator(Allocator alloc) {
		BC_ASSERT(m_curr_index==0,
				"Workspace set_allocator called while memory is still allocated");
		m_allocator.set_allocator(alloc);
	}

	Byte* allocate(std::size_t sz) {
		BC_ASSERT(m_curr_index + sz <= m_memptr_sz,
				"BC_Memory Allocation failure, attempting to allocate memory larger that workspace size");

		Byte* mem = m_memptr + m_curr_index;
		m_curr_index += sz;
		return mem;
	}

	void deallocate(Byte* memptr, std::size_t sz) {
		BC_ASSERT(memptr == (m_memptr + m_curr_index - sz),
				"BC_Memory Deallocation failure, attempting to deallocate memory out of order,"
				"\nWorkspace memory functions as a stack, deallocations must be in reverse order of allocations.");

		m_curr_index -= sz;
	}

	template<class T>
	T* allocate(std::size_t sz) {
		return reinterpret_cast<T*>(allocate(sz * sizeof(T)));
	}

	template<class T>
	void deallocate(T* memptr, std::size_t sz) {
		deallocate(reinterpret_cast<Byte*>(memptr), sz * sizeof(T));
	}

	~Workspace_Base(){
		BC_ASSERT(m_curr_index==0,
				"Workspace Destructor called while memory is still allocated, Memory Leak Detected");
		m_allocator.deallocate(m_memptr, m_memptr_sz);
	}
};
} //end of ns detail


/// An unsynced memory pool implemented as a stack.
/// Deallocation must happen in reverse order of deallocation.
/// This class is used with BC::Tensor_Base expressions to enable very fast allocations of temporaries.
template<class SystemTag, class ValueType=BC::allocators::Byte>
class Workspace {

	template<class, class>
	friend class Workspace;

	using ws_base_t = detail::Workspace_Base<SystemTag>;
	std::shared_ptr<ws_base_t> ws_ptr;

public:

	using system_tag = SystemTag;
	using value_type = ValueType;
	using propagate_on_container_copy_construction = std::false_type;

	template<class T>
	struct rebind {
		using other = Workspace<SystemTag,  T>;
	};

	Workspace(int sz=0)
	: ws_ptr(new ws_base_t(sz)) {}

	template<class T>
	Workspace(const Workspace<SystemTag, T>& ws) : ws_ptr(ws.ws_ptr) {}

	Workspace(const Workspace&)=default;
	Workspace(Workspace&&)=default;

	///Reserve an amount of memory in bytes.
	void reserve(std::size_t sz)  { ws_ptr->reserve(sz); }

	/// Delete all reserved memory, if memory is currently allocated an error is thrown.
	void free() { ws_ptr->free(); }

	size_t available_bytes() const {
		 return ws_ptr->available_bytes();
	}
	size_t allocated_bytes() const {
		 return ws_ptr->allocated_bytes();
	}
	size_t reserved_bytes() const {
		 return ws_ptr->reserved_bytes();
	}

	template<class Allocator>
	void set_allocator(Allocator alloc) {
		ws_ptr->set_allocator(alloc);
	}

	ValueType* allocate(std::size_t sz) {
		return ws_ptr->template allocate<ValueType>(sz);
	}
	void deallocate(ValueType* memptr, std::size_t sz) {
		ws_ptr->template deallocate<ValueType>(memptr, sz);
	}
};
}
}
}

#endif /* WORKSPACE_H_ */
