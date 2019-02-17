/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_CORE_OPERATIONS_BLAS_H_
#define BC_CORE_OPERATIONS_BLAS_H_

#include "Tags.h"

namespace BC   {
namespace oper {

//tags, no definition
template<class system_tag> struct transpose  {};
template<class system_tag> struct gemm : BLAS_Function {};
template<class system_tag> struct gemv : BLAS_Function {};
template<class system_tag> struct ger : BLAS_Function {};
template<class system_tag> struct dot : BLAS_Function {};

}
}



#endif /* BLAS_H_ */
