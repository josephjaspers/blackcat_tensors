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

template<class system_tag>
struct broadcasted_assign {
	broadcasted_assign() {
//		static_assert(false, "broadcasted_assign- should not be instantiated");
	}
};

template<class system_tag>
struct broadcasted_add_assign;

template<>
struct broadcasted_add_assign<BC::host_tag>
: linear_assignment_operation, beta_modifier<1>, alpha_modifier<1> {
        template<class lv, class rv>
		auto& operator ()(lv& l, rv r) const {
        	__BC_omp_atomic__
        	meta::bc_const_cast(l) += r;
        	return l;
        }
    };

template<class system_tag>
struct broadcasted_mul_assign;

template<>
struct broadcasted_mul_assign<host_tag> : assignment_operation {
	template<class lv, class rv>
	 auto operator ()(lv& l, rv r) const {
		__BC_omp_atomic__
		meta::bc_const_cast(l) *= r;
		return l;
	}
};

template<class system_tag>
struct broadcasted_sub_assign;

template<>
struct broadcasted_sub_assign<BC::host_tag> : linear_assignment_operation, beta_modifier<1>, alpha_modifier<-1> {
	template<class lv, class rv>
	 auto operator ()(lv& l, rv r) const {
		__BC_omp_atomic__
		meta::bc_const_cast(l) -= r;
		return l;
	}
};

template<class system_tag>
struct broadcasted_div_assign;

template<>
struct broadcasted_div_assign<BC::host_tag> : assignment_operation {
	template<class lv, class rv>
	 auto operator ()(lv& l, rv r) const {
		__BC_omp_atomic__
		meta::bc_const_cast(l) /= r;
		return l;
	}
};

#ifdef __CUDACC__


template<class system_tag>
struct broadcasted_add_assign;

template<>
struct broadcasted_add_assign<BC::device_tag>
: linear_assignment_operation, beta_modifier<1>, alpha_modifier<1> {
        template<class lv, class rv> BCINLINE
		auto& operator ()(lv& l, rv r) const {
        	atomicAdd(&meta::bc_const_cast(l), r);
        	return l;
        }
    };

template<class system_tag>
struct broadcasted_mul_assign;

template<>
struct broadcasted_mul_assign<BC::device_tag> : assignment_operation {
	template<class lv, class rv> BCINLINE
	 auto operator ()(lv& l, rv r) const {
    	atomicMul(&meta::bc_const_cast(l), r);
    	return l;
	}
};

template<class system_tag>
struct broadcasted_sub_assign;

template<>
struct broadcasted_sub_assign<BC::device_tag> : linear_assignment_operation, beta_modifier<1>, alpha_modifier<-1> {
	template<class lv, class rv> BCINLINE
	 auto operator ()(lv& l, rv r) const {
    	atomicSub(&meta::bc_const_cast(l), r);
    	return l;
	}
};

template<class system_tag>
struct broadcasted_div_assign;

template<>
struct broadcasted_div_assign<BC::device_tag> : assignment_operation {
	template<class lv, class rv> BCINLINE
	 auto operator ()(lv& l, rv r) const {
    	atomicDiv(&meta::bc_const_cast(l), r);
    	return l;
	}
};

#endif

}
}



#endif /* BINARY_BROADCASTED_H_ */
