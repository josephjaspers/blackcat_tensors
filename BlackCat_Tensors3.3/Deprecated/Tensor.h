/*
 * Tensor.h
 *
 *  Created on: Dec 30, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_N4_H
#define BC_TENSOR_N4_H

#include "BC_Tensor_Base/Tensor_Base.h"

namespace BC {

template<class scalar_t, class allocator_t>
using Scalar = Tensor_Base<internal::Array<0, scalar_t, allocator_t>>;
template<class scalar_t, class allocator_t>
using Vector = Tensor_Base<internal::Array<1, scalar_t, allocator_t>>;
template<class scalar_t, class allocator_t>
using Matrix = Tensor_Base<internal::Array<2, scalar_t, allocator_t>>;
template<class scalar_t, class allocator_t>
using Cube = Tensor_Base<internal::Array<3, scalar_t, allocator_t>>;

} //End Namespace BC

#endif /* MATRIX_H */
