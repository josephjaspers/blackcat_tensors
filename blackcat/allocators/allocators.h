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

namespace bc {
namespace allocators {

class Byte {};

template<class SystemTag, class ValueType>
class Allocator;

}

using allocators::allocator_traits;
using allocators::Allocator;

template<class ValueType>
using Basic_Allocator = allocators::Allocator<host_tag, ValueType>;

#ifdef __CUDACC__
template<class ValueType>
using Cuda_Allocator = allocators::Allocator<device_tag, ValueType>;

template<class ValueType>
using Cuda_Managed = allocators::Device_Managed<ValueType>;
#endif

} //end of namespace BC

//Assume All other Allocators may depend upon Allocator_Traits and Allocator
#include "polymorphic_allocator.h"
#include "stack_allocator.h"
#include "recycle_allocator.h"
#include "atomic_allocator.h"
#include "shared_allocator.h"


#endif
