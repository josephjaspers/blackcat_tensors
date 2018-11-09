/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BLACKCAT_COMPILERDEFINITIONS_H_
#define BLACKCAT_COMPILERDEFINITIONS_H_

namespace BC {

#define BLACKCAT_TENSORS_ASSERT_VALID                            //Ensures basic checks

#ifdef __CUDACC__
#define __BChd__ __host__ __device__
#define BLACKCAT_GPU_ENABLED
#else
#define __BChd__
#endif

#define __BCinline__ __BChd__  inline __attribute__((always_inline)) __attribute__((hot))
#define __BChot__   inline __attribute__((always_inline)) __attribute__((hot))

#define __BC_host_inline__ inline __attribute__((always_inline)) __attribute__((hot))

class BC_Type {}; //a type inherited by expressions and tensor_cores, it is used a flag and lacks a "genuine" implementation
class BC_Array {};
class BLAS_FUNCTION {};

}
#include <type_traits>
#include "Internal_Type_Traits.h"
#include "Utility/MetaTemplateFunctions.h"
#include "Utility/ShapeHierarchy.h"
#endif /* BLACKCAT_COMPILERDEFINITIONS_H_ */
