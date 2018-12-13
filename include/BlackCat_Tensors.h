/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BLACKCAT_TENSORS_H_
#define BLACKCAT_TENSORS_H_

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


// --------------------------------- constants --------------------------------- //

#ifdef __CUDACC__
namespace BC {
    static constexpr int CUDA_BASE_THREADS = 256;

    static int blocks(int size) {
        return 1 + (int)(size / CUDA_BASE_THREADS);
    }
    static int threads(int sz = CUDA_BASE_THREADS) {
        return sz > CUDA_BASE_THREADS ? CUDA_BASE_THREADS : sz;
    }
}
#endif




#include "Tensor_Base.h"
#include "Tensor_Aliases.h"

#endif /* BLACKCAT_TENSORS_H_ */
