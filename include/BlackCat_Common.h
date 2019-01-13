/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BLACKCAT_BLACKCAT_COMMON_H_
#define BLACKCAT_BLACKCAT_COMMON_H


#include <type_traits>
#include "evaluator/Evaluator.h"
#include "allocator/Allocator.h"
#include "random/Random.h"
#include "utility/Utility.h"
#include "blas/BLAS.h"
#include "streams/Streams.h"

namespace BC {

template<int>   class DISABLED;
template<class> class Tensor_Base;

class host_tag;
class device_tag;

class BC_Type {}; //a type inherited by expressions and tensor_cores, it is used a flag and lacks a "genuine" implementation
class BC_Array {};
class BC_Expr  {};
class BC_Temporary {};
class BLAS_FUNCTION {};

template<class T> static constexpr bool is_bc_type()    { return std::is_base_of<BC_Type, T>::value; }
template<class T> static constexpr bool is_array()      { return std::is_base_of<BC_Array, T>::value; }
template<class T> static constexpr bool is_expr()       { return std::is_base_of<BC_Expr, T>::value; }
template<class T> static constexpr bool is_temporary()  { return std::is_base_of<BC_Temporary, T>::value; }
template<class T> static constexpr bool is_blas_func()  { return std::is_base_of<BLAS_FUNCTION, T>::value; }

template<class internal_t>
auto make_tensor(internal_t internal) {
    return Tensor_Base<internal_t>(internal);
}

}

#endif /* BC_INTERNALS_BC_TENSOR_TENSOR_COMMON_H_ */
