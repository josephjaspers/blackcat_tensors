/*
 * Allocators.h
 *
 *  Created on: Dec 4, 2018
 *      Author: joseph
 */

#ifndef BLACKCAT_ALLOCATOR_ALLOCATOR_H_
#define BLACKCAT_ALLOCATOR_ALLOCATOR_H_

#include "basic_allocators.h"
#include "allocator_traits.h"

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

//Assume All other Allocators may depend upon Allocator_Traits and Allocator
#include "polymorphic_allocator.h"
#include "workspace.h"
#include "recycle_allocator.h"
#include "atomic_allocator.h"
#include "shared_allocator.h"


#endif
