/*
 * BlackCat_CompilerDefinitions.h
 *
 *  Created on: Jan 15, 2018
 *      Author: joseph
 */

#ifndef BLACKCAT_COMPILERDEFINITIONS_H_
#define BLACKCAT_COMPILERDEFINITIONS_H_

namespace BC {

static constexpr int BC_CPU_SINGLE_THREAD_THRESHOLD = 999;
static constexpr int CUDA_BASE_THREADS = 256;

#define BLACKCAT_TENSORS_ASSERT_VALID							//Ensures basic checks

#ifdef __CUDACC__
#define __BChd__ __host__ __device__
#define BLACKCAT_GPU_ENABLED
#else
#define __BChd__
#endif

#define __BCinline__ __BChd__  inline __attribute__((always_inline)) __attribute__((hot))
#define __BC_host_inline__ inline __attribute__((always_inline)) __attribute__((hot))
}


#include "BC_Utility/Determiners.h"
#include "BC_Utility/MetaTemplateFunctions.h"
#include "BC_Utility/ParameterPackModifiers.h"

#include "BC_Utility/Shape_Utility.h" //DEPENDENT UPON __BCinline__

#endif /* BLACKCAT_COMPILERDEFINITIONS_H_ */
