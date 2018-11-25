/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_INTERNALS_BC_TENSOR_TENSOR_COMMON_H_
#define BC_INTERNALS_BC_TENSOR_TENSOR_COMMON_H_


#include <type_traits>
#include "mathematics/CPU.h"
#include "mathematics/GPU.cu"
#include "allocators/Basic_Allocator.h"
#include "allocators/CUDA_Allocator.h"
#include "allocators/CUDA_Managed_Allocator.h"

namespace BC {
#define BC_ARRAY_ONLY(literal) static_assert(BC::et::is_array<internal_t>(), "BC Method: '" literal "' IS NOT SUPPORTED FOR EXPRESSIONS")
template<int>   class DISABLED;
template<class> class Tensor_Base;

class host_tag;
class device_tag;

#define BLACKCAT_TENSORS_ASSERT_VALID                            //Ensures basic checks

#ifdef __CUDACC__
	#define __BChd__ __host__ __device__
#else
	#define __BChd__
#endif

#define __BCinline__ __BChd__  inline __attribute__((always_inline)) __attribute__((hot))  //host_device inline
#define __BChot__   		   inline __attribute__((always_inline)) __attribute__((hot))  //device-only inline

class BC_Type {}; //a type inherited by expressions and tensor_cores, it is used a flag and lacks a "genuine" implementation
class BC_Array {};
class BLAS_FUNCTION {};


template<class internal_t>
auto make_tensor(internal_t internal) {
    return Tensor_Base<internal_t>(internal);
}

}

#endif /* BC_INTERNALS_BC_TENSOR_TENSOR_COMMON_H_ */
