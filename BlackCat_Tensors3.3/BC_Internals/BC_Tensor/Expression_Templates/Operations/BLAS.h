/*
 * BLAS.h
 *
 *  Created on: Sep 5, 2018
 *      Author: joseph
 */

#ifndef BLAS_H_
#define BLAS_H_

namespace BC {
namespace internal {
namespace opers {

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
