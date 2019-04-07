/*
 * Allocator_Traits.h
 *
 *  Created on: Jan 1, 2019
 *      Author: joseph
 */

#ifndef BC_ALLOCATOR_ALLOCATOR_TRAITS_H_
#define BC_ALLOCATOR_ALLOCATOR_TRAITS_H_

#include <memory>

namespace BC {

class host_tag;
class device_tag;

namespace allocator {
namespace traits {

template<class alloc, class enabler=void>
struct system_tag_of : std::false_type {
	using type = host_tag;
};

template<class alloc>
struct system_tag_of<alloc, std::enable_if_t<!std::is_void<typename alloc::system_tag>::value>>
: std::true_type {
	using type = typename alloc::system_tag;
};

}

template<class Allocator>
struct allocator_traits : std::allocator_traits<Allocator> {
	using system_tag = typename traits::system_tag_of<Allocator>::type;
};

}
}



#endif /* ALLOCATOR_TRAITS_H_ */
