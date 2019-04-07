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


#ifdef __CUDACC__
template<class SystemTag, class ValueType>
using implementation = std::conditional_t<std::is_same<host_tag, SystemTag>::value, Host<ValueType>, Device<ValueType>>;
#else
template<class SystemTag, class ValueType>
using implementation = std::conditional_t<std::is_same<host_tag, SystemTag>::value, Host<ValueType>, void>;
#endif



}

//Push allocator_traits into the BC namespace
template<class Allocator_Type>
using allocator_traits = allocator::allocator_traits<Allocator_Type>;

#ifdef __CUDACC__
template<class value_type>
using Cuda_Allocator = allocator::Device<value_type>;

template<class value_type>
using Cuda_Managed = allocator::Device_Managed<value_type>;
#endif

template<class value_type>
using Basic_Allocator = allocator::Host<value_type>;

} //ns BC

// --- Include "fancy allocators" below here --- //
// Fancy allocators are included below as they more depend upon the
//	 default allocator and/or BC::Allocator_Traits

#include "fancy/Polymorphic_Allocator.h"
#include "fancy/Workspace.h"
#include "fancy/Scalar_Recycled_Workspace.h"


#endif
