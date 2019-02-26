/*
 * Allocators.h
 *
 *  Created on: Dec 4, 2018
 *      Author: joseph
 */

#ifndef BC_ALLOCATOR_ALLOCATOR_H_
#define BC_ALLOCATOR_ALLOCATOR_H_

#include "Host.h"
#include "Device.h"
#include "Device_Managed.h"

namespace BC {


class host_tag;
class device_tag;

namespace allocator {

//#ifdef __CUDACC__
//	template<class system_tag, class value_type>
//	using implementation =
//			std::conditional_t<
//				std::is_same<host_tag, system_tag>::value,
//					Host<value_type>,
//					Device<value_type>>;
//
//#else
//	template<
//		class system_tag, class value_type,
//		class=std::enable_if<std::is_same<system_tag, host_tag>::value>
//	>
//	using implementation = Host<value_type>;
//#endif

}
}


//Allocator_traits.h depends upon allocator::implementation
#include "Allocator_Traits.h"

namespace BC {

//Push allocator_traits into the BC namespace
template<class Allocator_Type>
using allocator_traits = allocator::allocator_traits<Allocator_Type>;

}

#endif
