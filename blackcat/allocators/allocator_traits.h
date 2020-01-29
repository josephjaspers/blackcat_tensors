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
class device_tag;

namespace allocators {
namespace detail {

	template<class T>
	using query_managed_memory = bc::traits::truth_type<T::managed_memory>;

}

template<class Allocator>
struct allocator_traits : std::allocator_traits<Allocator>
{
	using system_tag = bc::traits::conditional_detected_t<
		bc::traits::query_system_tag, Allocator, host_tag>;

	using is_managed_memory_t = bc::traits::conditional_detected_t<
		detail::query_managed_memory, Allocator, std::false_type>;

	static constexpr bool is_managed_memory = is_managed_memory_t::value;
};

}
}



#endif /* ALLOCATOR_TRAITS_H_ */
