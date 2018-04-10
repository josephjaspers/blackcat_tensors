/* 
 * File:   BLACKCAT_EXPLICIT_INSTANTIATION.h
 * Author: joseph
 *
 * Created on July 31, 2017, 7:54 PM
 */

#ifndef BLACKCAT_EXPLICIT_INSTANTIATION_H
#define BLACKCAT_EXPLICIT_INSTANTIATION_H


#include "LinearAlgebraRoutines.h"
template class Tensor_Operations<double>;
template class Tensor_Operations<float>;
template class Tensor_Operations<unsigned>;
template class Tensor_Operations<signed>;

#include "Tensor.h"
template class Tensor<double>;
template class Tensor<float>;
template class Tensor<unsigned>;
template class Tensor<signed>;

#include "Scalar.h"
template class Scalar<double>;
template class Scalar<unsigned>;
template class Scalar<signed>;
template class Scalar<float>;

#include "Matrix.h"
template class Matrix<double>;
template class Matrix<unsigned>;
template class Matrix<signed>;
template class Matrix<float>;

#include "Vector.h"
template class Vector<double>;
template class Vector<unsigned>;
template class Vector<signed>;
template class Vector<float>;

#endif /* BLACKCAT_EXPLICIT_INSTANTIATION_H */

