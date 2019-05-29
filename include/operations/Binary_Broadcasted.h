/*
 * Binary_Broadcasted.h
 *
 *  Created on: Feb 19, 2019
 *      Author: joseph
 */

#ifndef BC_CORE_BINARY_BROADCASTED_H_
#define BC_CORE_BINARY_BROADCASTED_H_

#include <type_traits>
#include <cmath>
#include "Tags.h"
#include <mutex>
#include <iostream>

#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#endif

namespace BC {
namespace oper {

#define BC_BIN_OP(...)\
	template<class Lv, class Rv>\
	BCINLINE \
	static Lv& apply (Lv& lv, Rv&& rv) {\
		__VA_ARGS__;\
	}\
	template<class Lv, class Rv>\
	BCINLINE Lv& operator () (Lv& lv, Rv&& rv) const {\
		return apply(lv, rv);\
	}\
	template<class Lv, class Rv>\
	BCINLINE Lv& operator () (Lv& lv, Rv&& rv) {\
		return apply(lv, rv);\
	}


template<class system_tag>
struct broadcasted_add_assign;

template<>
struct broadcasted_add_assign<BC::host_tag>
: linear_assignment_operation, beta_modifier<1>, alpha_modifier<1> {
	BC_BIN_OP(
        	BC_omp_atomic__
        	lv += rv;
			return lv;
		)
    };

template<class system_tag>
struct broadcasted_mul_assign;

template<>
struct broadcasted_mul_assign<host_tag> : assignment_operation {
	BC_BIN_OP(
		BC_omp_atomic__
		lv *= rv;
		return lv;
	)
};

template<class system_tag>
struct broadcasted_sub_assign;

template<>
struct broadcasted_sub_assign<BC::host_tag> : linear_assignment_operation, beta_modifier<1>, alpha_modifier<-1> {
	BC_BIN_OP(
		BC_omp_atomic__
		lv -= rv;
		return lv;
	)
};

template<class system_tag>
struct broadcasted_div_assign;

template<>
struct broadcasted_div_assign<BC::host_tag> : assignment_operation {
	BC_BIN_OP(
		BC_omp_atomic__
		lv /= rv;
		return lv;
	)
};

#ifdef __CUDACC__


template<class system_tag>
struct broadcasted_add_assign;

template<>
struct broadcasted_add_assign<BC::device_tag>
: linear_assignment_operation, beta_modifier<1>, alpha_modifier<1> {
	BC_BIN_OP(
        	atomicAdd(&lv, rv);
        	return lv;
        )
    };


template<class system_tag>
struct broadcasted_sub_assign;

template<>
struct broadcasted_sub_assign<BC::device_tag> : linear_assignment_operation, beta_modifier<1>, alpha_modifier<-1> {
	BC_BIN_OP(
    	atomicAdd(&lv, -rv); //atomicSub doesn't support floats
    	return lv;
	)
};

template<class system_tag>
struct broadcasted_div_assign;

template<>
struct broadcasted_div_assign<BC::device_tag> : assignment_operation {
	BC_BIN_OP(
		static_assert(
				std::is_same<void, Lv>::value,
				"BLACKCAT_TENSORS: broadcasted-reduction div-assign is currently not available on the GPU");
    	return lv;
	)
};

template<class system_tag>
struct broadcasted_mul_assign;

template<>
struct broadcasted_mul_assign<BC::device_tag> : assignment_operation {
	BC_BIN_OP(
		static_assert(
				std::is_same<void, Lv>::value,
				"BLACKCAT_TENSORS: broadcasted-reduction mul-assign is currently not available on the GPU");
    	return lv;
	)
};


#endif

}
}


#undef BC_BIN_OP
#endif /* BINARY_BROADCASTED_H_ */
