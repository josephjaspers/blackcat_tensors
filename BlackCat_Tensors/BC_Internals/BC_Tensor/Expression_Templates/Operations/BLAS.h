/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BLAS_H_
#define BLAS_H_

namespace BC {
namespace internal {
namespace oper {

template<class mathlib> struct transpose;
template<class mathlib> struct gemm;
template<class mathlib> struct gemv;
template<class mathlib> struct ger;
template<class mathlib> struct dot;
template<int x,class mathlib> struct conv;



}
}
}



#endif /* BLAS_H_ */
