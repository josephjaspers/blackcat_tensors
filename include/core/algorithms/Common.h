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

#ifdef BC_CPP17
    #ifndef BC_CPP17_EXECUTION
        #define BC_CPP17_EXECUTION std::execution::par,
    #endif
#endif


#endif /* COMMON_H_ */
