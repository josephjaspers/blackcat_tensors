/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_CORE_OPERATIONS_BLAS_H_
#define BC_CORE_OPERATIONS_BLAS_H_

#include "Operation_Traits.h"
#include "Tags.h"

namespace BC   {
namespace oper {
namespace detail {

template<class DerivedOperation>
struct matmul_forward_backward {

	template<class Lv, class Rv>
	auto forward_propagate(Lv&& lv_, Rv&& rv_) const {
		auto&& lv = BC::oper::operation_traits<Lv>::select_on_forward_propagate(lv_);
		auto&& rv = BC::oper::operation_traits<Rv>::select_on_forward_propagate(rv_);
		return lv * rv;
	}

	template<class Delta, class Lv_x, class Rv_x, class Lv, class Rv>
	void backward_propagate(Delta&& dy, Lv_x&& lv_x, Rv_x&& rv_x, Lv&& lv, Rv&& rv) const {
		lv.backward_propagate(dy * rv_x.transpose());
		rv.backward_propagate(lv_x * dy.transpose());
	}
};



}


//tags, no definition
template<class system_tag> struct transpose  { };
template<class system_tag>

struct gemm:
		BLAS_Function,
		detail::matmul_forward_backward<gemm<system_tag>> {
	static constexpr int tensor_dimension = 2;
};
template<class system_tag>
struct gemv:
		BLAS_Function,
		detail::matmul_forward_backward<gemm<system_tag>> {
	static constexpr int tensor_dimension = 1;
};

template<class system_tag> struct ger : BLAS_Function  { static constexpr int tensor_dimension = 2; };
template<class system_tag> struct dot : BLAS_Function  { static constexpr int tensor_dimension = 0; };

}
}



#endif /* BLAS_H_ */
