/*
 * BlackCat_CompilerDefinitions.h
 *
 *  Created on: Jan 15, 2018
 *      Author: joseph
 */

#ifndef BLACKCAT_COMPILERDEFINITIONS_H_
#define BLACKCAT_COMPILERDEFINITIONS_H_

namespace BC {

#define BLACKCAT_TENSORS_ASSERT_VALID							//Ensures basic checks

#ifdef __CUDACC__
#define __BChd__ __host__ __device__
#define BLACKCAT_GPU_ENABLED
#else
#define __BChd__
#endif

#define __BCinline__ __BChd__  inline __attribute__((always_inline)) __attribute__((hot))
#define __BC_host_inline__ inline __attribute__((always_inline)) __attribute__((hot))

class BC_Type {}; //a type inherited by expressions and tensor_cores, it is used a flag and lacks a "genuine" implementation
class BC_Core {};
class BLAS_FUNCTION {};

}
#include "BC_Utility/Determiners.h"
#include "BC_Utility/MetaTemplateFunctions.h"
#include "BC_Utility/ParameterPackModifiers.h"
#include "BC_Utility/ShapeHierarchy.h"
#endif /* BLACKCAT_COMPILERDEFINITIONS_H_ */
