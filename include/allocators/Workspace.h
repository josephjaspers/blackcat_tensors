/*
 * Stack_Allocator.h
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
namespace detail {

template<class SystemTag>
class Stack_Allocator_Base {

	using system_tag = SystemTag;

	std::size_t m_data_sz=0;
	std::size_t m_curr_index=0;

	Byte* m_data=nullptr;

	static Polymorphic_Allocator<SystemTag, BC::allocators::Byte>& get_default_allocator() {
		static Polymorphic_Allocator<SystemTag, BC::allocators::Byte> default_allocator;
		return default_allocator;
	}

	Polymorphic_Allocator<SystemTag, Byte> m_allocator = get_default_allocator();

public:
	Stack_Allocator_Base(std::size_t sz=0) : m_data_sz(sz){
		if (sz)
			m_data = m_allocator.allocate(sz);
	}
	void reserve(std::size_t sz)  {
		if (sz == 0 || (m_data_sz - m_curr_index) > sz) {
			return;
		}

		if (!(m_curr_index==0)){
			std::cout << "BC_Memory Allocation failure: \n" <<
					"\tcurrent_stack_index == " << m_curr_index << " out of " << m_data_sz << \
					"attempting to reserve " << sz  << " bytes " << std::endl;
		}

		BC_ASSERT(m_curr_index==0,
				"Stack_Allocator reserve called while memory is still allocated");

		if (m_data_sz < sz) {
			if (m_data_sz > 0) {
				m_allocator.deallocate(m_data, m_data_sz);
			}
			m_data = m_allocator.allocate(sz);
			m_data_sz = sz;
		}
	}
	void free() {
		BC_ASSERT(m_curr_index==0,
				"Stack_Allocator free called while memory is still allocated");
		m_allocator.deallocate(m_data, m_data_sz);
		m_data_sz = 0;
		m_data = nullptr;
	}

	size_t available_bytes() const {
		 return m_data_sz - m_curr_index;
	}
	size_t allocated_bytes() const {
		 return m_curr_index;
	}
	size_t reserved_bytes() const {
		 return m_data_sz;
	}

	template<class Allocator>
	void set_allocator(Allocator alloc) {
		BC_ASSERT(m_curr_index==0,
				"Stack_Allocator set_allocator called while memory is still allocated");
		m_allocator.set_allocator(alloc);
	}

	Byte* allocate(std::size_t sz) {
		if (m_curr_index + sz > m_data_sz) {
			std::cout << "BC_Memory Allocation failure: \n" <<
					"\tcurrent_stack_index == " << m_curr_index << " out of " << m_data_sz
					<<"\n\t attempting to allocate " << sz << " bytes, error: " << (m_curr_index + sz <= m_data_sz) << std::endl;
		}

		BC_ASSERT(!(m_curr_index + sz > m_data_sz),
				"BC_Memory Allocation failure, attempting to allocate memory larger that workspace size");

		Byte* mem = m_data + m_curr_index;
		m_curr_index += sz;
		return mem;
	}

	void deallocate(Byte* data, std::size_t sz) {
		BC_ASSERT(m_curr_index!=0,
				"BC_Memory Deallocation failure, attempting to deallocate already deallocated memory.");

		BC_ASSERT(data == (m_data + m_curr_index - sz),
				"BC_Memory Deallocation failure, attempting to deallocate memory out of order,"
				"\nStack_Allocator memory functions as a stack, deallocations must be in reverse order of allocations.");


		m_curr_index -= sz;
	}


	void force_deallocate() {
		m_curr_index = 0;
	}

	template<class T>
	T* allocate(std::size_t sz) {
		return reinterpret_cast<T*>(allocate(sz * sizeof(T)));
	}

	template<class T>
	void deallocate(T* data, std::size_t sz) {
		deallocate(reinterpret_cast<Byte*>(data), sz * sizeof(T));
	}

	~Stack_Allocator_Base(){
		BC_ASSERT(m_curr_index==0,
				"Stack_Allocator Destructor called while memory is still allocated, Memory Leak Detected");
		m_allocator.deallocate(m_data, m_data_sz);
	}
};
} //end of ns detail


/// An unsynced memory pool implemented as a stack.
/// Deallocation must happen in reverse order of deallocation.
/// This class is used with BC::Tensor_Base expressions to enable very fast allocations of temporaries.
template<class SystemTag, class ValueType=BC::allocators::Byte>
class Stack_Allocator {

	template<class, class>
	friend class Stack_Allocator;

	using ws_base_t = detail::Stack_Allocator_Base<SystemTag>;
	std::shared_ptr<ws_base_t> ws_ptr;

public:

	using system_tag = SystemTag;
	using value_type = ValueType;
	using propagate_on_container_copy_construction = std::false_type;

	template<class T>
	struct rebind {
		using other = Stack_Allocator<SystemTag,  T>;
	};

	Stack_Allocator(int sz=0)
	: ws_ptr(new ws_base_t(sz)) {}

	template<class T>
	Stack_Allocator(const Stack_Allocator<SystemTag, T>& ws) : ws_ptr(ws.ws_ptr) {}

	Stack_Allocator(const Stack_Allocator&)=default;
	Stack_Allocator(Stack_Allocator&&)=default;

	///Reserve an amount of memory in bytes.
	void reserve(std::size_t sz)  { ws_ptr->reserve(sz * sizeof(value_type)); }

	/// Delete all reserved memory, if memory is currently allocated an error is thrown.
	void free() { ws_ptr->free(); }

	void force_deallocate() {
		ws_ptr->force_deallocate();
	}

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
	void deallocate(ValueType* data, std::size_t sz) {
		ws_ptr->template deallocate<ValueType>(data, sz);
	}
};

}
}


#endif /* WORKSPACE_H_ */
