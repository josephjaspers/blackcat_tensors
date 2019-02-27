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

//Push allocator_traits into the BC namespace
template<class Allocator_Type>
using allocator_traits = allocator::allocator_traits<Allocator_Type>;

}

#endif
