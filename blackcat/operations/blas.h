/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_CORE_OPERATIONS_BLAS_H_
#define BC_CORE_OPERATIONS_BLAS_H_

#include "operation_traits.h"
#include "tags.h"

namespace bc   {
namespace oper {

template<class system_tag>
struct transpose  {};

template<class system_tag>
struct gemm: BLAS_Function {
	static constexpr int tensor_dim = 2;
};

template<class system_tag>
struct gemv: BLAS_Function {
	static constexpr int tensor_dim = 1;
};

template<class system_tag>
struct ger : BLAS_Function  {
	static constexpr int tensor_dim = 2;
};

template<class system_tag>
struct dot : BLAS_Function  {
	static constexpr int tensor_dim = 0;
};

}
}



#endif /* BLAS_H_ */
