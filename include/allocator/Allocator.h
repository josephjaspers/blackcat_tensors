/*
 * Allocators.h
 *
 *  Created on: Dec 4, 2018
 *      Author: joseph
 */

#ifndef BC_ALLOCATOR_ALLOCATOR_H_
#define BC_ALLOCATOR_ALLOCATOR_H_

#include <memory>

namespace BC {
namespace allocator {

template<class derived>
struct AllocatorBase {
	AllocatorBase() {
	}
};

template<class alloc, class enabler=void>
struct system_tag_of : std::false_type { using type = host_tag; };

template<class alloc>
struct system_tag_of<alloc, std::enable_if_t<!std::is_void<typename alloc::system_tag>::value>>
: std::false_type { using type = typename alloc::system_tag; };



template<class Allocator>
struct allocator_traits : std::allocator_traits<Allocator> {
	using system_tag = typename system_tag_of<Allocator>::type;
};

template<class allocator, class=void>
struct has_system_tag : system_tag_of<allocator> {};

} //end of namespace allocator

//Push allocator_traits into the BC namespace
template<class Allocator_Type>
using allocator_traits = allocator::allocator_traits<Allocator_Type>;

} //end of namespace BC


#include "Host.h"
#include "Device.h"
#include "Device_Managed.h"

namespace BC {

class host_tag;
class device_tag;

namespace allocator {

#ifdef __CUDACC__
	template<class system_tag, class value_type>
	using implementation =
			std::conditional_t<
				std::is_same<host_tag, system_tag>::value,
					Host<value_type>,
					Device<value_type>>;

#else
	template<
		class system_tag, class value_type,
		class=std::enable_if<std::is_same<system_tag, host_tag>::value>
	>
	using implementation = Host<value_type>;
#endif




}
}

#endif
