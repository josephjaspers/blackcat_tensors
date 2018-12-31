/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BLACKCAT_BLACKCAT_TENSORS_H_
#define BLACKCAT_BLACKCAT_TENSORS_H_

/*
 * This file defines all global macros and compilation switches.
 */

// --------------------------------- compile options --------------------------------- //

//#define BC_NO_OPENMP                    //Disables automatic multi-threading of element-wise operations (if openmp is linked)
#define BC_CPP17                          //Enables C++17 features
//#define BC_DISABLE_TEMPORARIES		  //Disables the creation of temporaries in expressions
//#define BC_EXECUTION_POLICIES           //Enables execution policies
//#define NDEBUG 		                  //Disables runtime checks
//BC_CPP17_EXECUTION std::execution::par  //defines default execution as parallel
//BC_CPP17_EXECUTION std::execution::seq  //defines default execution as sequential
//#define BC_TREE_OPTIMIZER_DEBUG	      //enables print statements for the tree evaluator. (For developers)


// --------------------------------- override macro-options --------------------------------- //
//#define BC_INLINE_OVERRIDE <compiler_attribute>       //overloads the default inline attribute
//#define BC_SIZE_T_OVERRIDE  <integer_type>			//overloads the default size_t (default is signed int)

// --------------------------------- inline macros -----------------------------------------//

#ifdef __CUDACC__
	#define __BChd__ __host__ __device__
#else
	#define __BChd__
#endif

#ifdef BC_INLINE_OVERRIDER
#define __BCinline__ __BChd__  BC_INLINE_OVERRIDER
#else
#define __BCinline__ __BChd__  inline __attribute__((always_inline)) __attribute__((hot))  //host_device inline
#endif

#define __BChot__   		   inline __attribute__((always_inline)) __attribute__((hot))  //device-only inline

// --------------------------------- module body macro --------------------------------- //


#ifdef __CUDACC__	//------------------------------------------|

#define BC_DEFAULT_MODULE_BODY(namespace_name)					\
																\
namespace BC { 													\
																\
class host_tag;													\
class device_tag;												\
																\
namespace namespace_name {									   	\
																\
	template<class system_tag>									\
	using implementation =										\
			std::conditional_t<									\
				std::is_same<host_tag, system_tag>::value,		\
					Host,										\
					Device>;									\
																\
	}															\
}

#else

#define BC_DEFAULT_MODULE_BODY(namespace_name)					 \
																 \
namespace BC { 													 \
																 \
class host_tag;													 \
class device_tag;												 \
																 \
namespace namespace_name {										 \
																 \
	template<													 \
		class system_tag,										 \
		class=std::enable_if<std::is_same<system_tag, host_tag>::value> \
	>															 \
	using implementation = Host;								 \
																 \
	}															 \
}
#endif

// --------------------------------- openmp macros --------------------------------- //

#if defined(_OPENMP) && !defined(BC_NO_OPENMP)
	#define __BC_omp_for__ _Pragma("omp parallel for")
	#define __BC_omp_bar__ _Pragma("omp barrier")
#else
	#define __BC_omp_for__
	#define __BC_omp_bar__
#endif


//#define BC_SIZE_T_OVERRIDE unsigned
// --------------------------------- constants --------------------------------- //

namespace BC {

#ifndef BC_SIZE_T_OVERRIDE
using  size_t   = int;
#else
using  size_t   = BC_SIZE_T_OVERRIDE;
#endif

#ifndef BC_DIM_T_OVERRIDE
using dim_t = int;
#else
using dim_t = BC_DIM_T_OVERRIDE
#endif

static constexpr  BC::size_t MULTITHREAD_THRESHOLD = 16384;

#ifdef __CUDACC__
    static constexpr BC::size_t CUDA_BASE_THREADS = 256;

    static  BC::size_t blocks(int size) {
        return 1 + (int)(size / CUDA_BASE_THREADS);
    }
    static  BC::size_t threads(int sz = CUDA_BASE_THREADS) {
        return sz > CUDA_BASE_THREADS ? CUDA_BASE_THREADS : sz;
    }
#endif

}

#include "BlackCat_Common.h"
#include "Tensor_Base.h"
#include "Tensor_Aliases.h"

#endif /* BLACKCAT_TENSORS_H_ */
