/*
 * common.h
 *
 *  Created on: Nov 24, 2018
 *      Author: joseph
 */

#ifndef BC_ALGORITHM_COMMON_H_
#define BC_ALGORITHM_COMMON_H_

#ifdef BC_CPP17
	#define BC_DEF_IF_CPP17(code) code
#else
	#define BC_DEF_IF_CPP17(code)
#endif

//multithreaded algorithms by default
//#define BC_ALG_SINGLE_PARALLEL_DEFAULT
#ifdef BC_CPP17
    #ifdef BC_ALG_SINGLE_PARALLEL_DEFAULT
        #define BC_CPU_ALGORITHM_EXECUTION std::execution::seq,
    #else
        #define BC_CPU_ALGORITHM_EXECUTION std::execution::par,
    #endif
#endif


#endif /* COMMON_H_ */
