/*
 * Allocator_Traits.h
 *
 *  Created on: Jan 1, 2019
 *      Author: joseph
 */

#ifndef BC_ALLOCATOR_ALLOCATOR_TRAITS_H_
#define BC_ALLOCATOR_ALLOCATOR_TRAITS_H_

#include <memory>

namespace bc {

class host_tag;

namespace allocators {

template<class Allocator>
struct allocator_traits : std::allocator_traits<Allocator>
{
	using system_tag = bc::traits::conditional_detected_t<
		bc::traits::query_system_tag, Allocator, host_tag>;
};

}
}

#endif /* ALLOCATOR_TRAITS_H_ */
