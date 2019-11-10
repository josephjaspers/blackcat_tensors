/*
 * Memory_Buffer.h
 *
 *  Created on: Nov 4, 2019
 *      Author: joseph
 */

#ifndef MEMORY_BUFFER_H_
#define MEMORY_BUFFER_H_

template<class T, class Allocator>
class Memory_Buffer {

	using value_type = T;
	using allocator_type = Allocator;
	using system_tag = typename BC::allocator_traits<Allocator>::system_tag;

	std::unique_ptr<T> m_data;

	[[no_unique_address]]
	 Allocator m_allocator;
};



#endif /* MEMORY_BUFFER_H_ */
