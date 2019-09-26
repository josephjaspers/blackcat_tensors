/*
 * Allocators.h
 *
 *  Created on: Dec 4, 2018
 *      Author: joseph
 */

#ifndef BLACKCAT_ALLOCATOR_ALLOCATOR_H_
#define BLACKCAT_ALLOCATOR_ALLOCATOR_H_

#include "Basic_Allocators.h"
#include "Allocator_Traits.h"

namespace BC {
namespace allocators {

class Byte {};
template<class SystemTag, class ValueType>
class Allocator;

}

using allocators::allocator_traits;
using allocators::Allocator;

template<class value_type>
using Basic_Allocator = allocators::Allocator<host_tag, value_type>;

#ifdef __CUDACC__
template<class value_type>
using Cuda_Allocator = allocators::Allocator<device_tag, value_type>;

template<class value_type>
using Cuda_Managed = allocators::Device_Managed<value_type>;
#endif


} //end of namespace BC

// --- Include "fancy allocators" below here --- //
//	 Fancy Allocators depend on allocators_traits/basic_allocator
#include "fancy/Polymorphic_Allocator.h"
#include "fancy/Workspace.h"
#include "fancy/Recycle_Allocator.h"
#include "fancy/Atomic_Allocator.h"
#include "fancy/Shared_Allocator.h"

namespace BC {
namespace allocators {

template<class Allocator>
using Shared_Atomic_Allocator = fancy::Shared_Allocator<fancy::Atomic_Allocator<Allocator>>;

}
}


#endif
