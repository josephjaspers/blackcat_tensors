/*
 * Allocators.h
 *
 *  Created on: Dec 4, 2018
 *      Author: joseph
 */

#ifndef BC_ALLOCATOR_ALLOCATOR_H_
#define BC_ALLOCATOR_ALLOCATOR_H_

namespace BC {
namespace allocator {

template<class derived>
struct AllocatorBase {
	AllocatorBase() {
        static_assert(std::is_trivially_copy_constructible<derived>::value, "BC ALLOCATORS TYPES MUST BE TRIVIALLY COPYABLE");
        static_assert(std::is_trivially_copyable<derived>::value, "BC ALLOCATORS MUST BE TRIVIALLY COPYABLE");
	}
};

template<class custom_allocator, class _system_tag=host_tag>
struct CustomAllocator
		: custom_allocator,
		  AllocatorBase<CustomAllocator<custom_allocator>> {
			using system_tag = _system_tag;
		};
}
}


#include "Host.h"
#include "Device.h"
#include "Device_Managed.h"

namespace BC {

class host_tag;
class device_tag;

namespace allocator {

#ifdef __CUDACC__
	template<class system_tag, class scalar_t>
	using implementation =
			std::conditional_t<
				std::is_same<host_tag, system_tag>::value,
					Host<scalar_t>,
					Device<scalar_t>>;

#else
	template<
		class system_tag, class scalar_t,
		class=std::enable_if<std::is_same<system_tag, host_tag>::value>
	>
	using implementation = Host<scalar_t>;
#endif




}
}

#endif
