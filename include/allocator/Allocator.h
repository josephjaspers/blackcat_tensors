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
#include "Allocator_Traits.h"

namespace BC {
namespace allocator {

class Byte {};

}

template<class Allocator_Type>
using allocator_traits = allocator::allocator_traits<Allocator_Type>;

template<class value_type>
using Basic_Allocator = allocator::Host<value_type>;


#ifdef __CUDACC__
template<class value_type>
using Cuda_Allocator = allocator::Device<value_type>;

template<class value_type>
using Cuda_Managed = allocator::Device_Managed<value_type>;

template<class system_tag, class value_type>
using Allocator = std::conditional_t<std::is_same<system_tag, host_tag>::value,
										Basic_Allocator<value_type>, Cuda_Allocator<value_type>>;

#else
template<class system_tag, class value_type>
using Allocator = Basic_Allocator<
		std::enable_if_t<
			std::is_same<system_tag, host_tag>::value,
			value_type>>;

#endif



} //end of namespace BC

// --- Include "fancy allocators" below here --- //
//	 Fancy Allocators depend on allocators_traits/basic_allocator
#include "fancy/Polymorphic_Allocator.h"
#include "fancy/Workspace.h"


#endif
