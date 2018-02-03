/*
 * BlackCat_CompilerDefinitions.h
 *
 *  Created on: Jan 15, 2018
 *      Author: joseph
 */

#ifndef BLACKCAT_COMPILERDEFINITIONS_H_
#define BLACKCAT_COMPILERDEFINITIONS_H_

namespace BC {



//#define DISABLE_BLACKCAT_DEBUG_CHECKS


#ifdef __CUDACC__
#define __BC_gcpu__ __host__ __device__
static constexpr int CUDA_BASE_THREADS = 256;
#define BLACKCAT_GPU_ENABLED
#else
#define __BC_gcpu__
#endif
}
//Ifndef __CUDACC__ add: -x c-header cu
//to the compiler command line --> this converts all files that would normally be interperted by the NVCC to header.files
#endif /* BLACKCAT_COMPILERDEFINITIONS_H_ */
