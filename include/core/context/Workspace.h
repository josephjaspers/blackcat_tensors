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
namespace context {

template<class SystemTag>
class Workspace {

	using system_tag = SystemTag;

	BC::size_t m_memptr_sz=0;
	BC::size_t m_curr_index=0;

	Byte* m_memptr=nullptr;

	static Polymorphic_Allocator<Byte, SystemTag> default_allocator;

	Polymorphic_Allocator<Byte, SystemTag> m_allocator =
			Polymorphic_Allocator<Byte, SystemTag>(default_allocator);

public:
	template<class Allocator>
	static void set_default_allocator(Allocator& alloc) {
		default_allocator.set_allocator(alloc);
	}
	template<class Allocator>
	static void set_default_allocator(const Allocator& alloc) {
		default_allocator.set_allocator(alloc);
	}


	Workspace(std::size_t sz=0) : m_memptr_sz(sz){
		if (sz)
			m_memptr = m_allocator.allocate(sz);
	}
	void reserve(std::size_t sz)  {
		BC_ASSERT(m_curr_index==0,
				"Workspace reserve called while memory is still allocated");

		if (!(m_memptr_sz > sz)) {
			m_allocator.deallocate(m_memptr, m_memptr_sz);
			m_memptr = m_allocator.allocate(sz);

		}
	}
	template<class Allocator>
	void set_allocator(Allocator alloc) {
		BC_ASSERT(m_curr_index==0,
				"Workspace set_allocator called while memory is still allocated");
		m_allocator.set_allocator(alloc);
	}

	Byte* allocate(std::size_t sz) {
		BC_ASSERT(m_curr_index + sz < m_memptr_sz,
				"BC_Memory Allocation failure, attempting to allocate memory larger that workspace size");

		Byte* mem = m_memptr + m_curr_index;
		m_curr_index += sz;
		return mem;
	}
	void deallocate(Byte* memptr, std::size_t sz) {
		BC_ASSERT(memptr == m_memptr + m_curr_index - sz,
				"BC_Memory Deallocation failure, attempting to deallocate memory out of order,"
				"\nWorkspace memory functions as a stack, deallocations must be in reverse order of allocations.");

		m_curr_index -= sz;
	}

	~Workspace(){
		BC_ASSERT(m_curr_index==0,
				"Workspace Destructor called while memory is still allocated, Memory Leak Detected");
		m_allocator.deallocate(m_memptr, m_memptr_sz);
	}



};

}
}




#endif /* WORKSPACE_H_ */
