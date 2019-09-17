/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */


#ifndef BLACKCAT_TENSOR_ALIASES_H_
#define BLACKCAT_TENSOR_ALIASES_H_

#include "Tensor_Common.h"
#include "Tensor_Base.h"
#include "../BlackCat_Allocator.h"

namespace BC {
namespace tensors {
namespace detail{

template<class T>
using default_allocator = BC::Allocator<default_system_tag_t, T>;

template<int X>
using default_shape = BC::Shape<X>;

} //end of ns detail
} //end of ns tensors

template<int dimension, class ValueType, class Allocator=tensors::detail::default_allocator<ValueType>>
using Tensor = BC::tensors::Tensor_Base<BC::tensors::exprs::Array<BC::Shape<dimension>, ValueType, Allocator>>;

template<class ValueType, class Allocator = tensors::detail::default_allocator<ValueType>>
using Scalar = Tensor<0, ValueType, Allocator>;

template<class ValueType, class Allocator = tensors::detail::default_allocator<ValueType>>
using Vector = Tensor<1, ValueType, Allocator>;

template<class ValueType, class Allocator = tensors::detail::default_allocator<ValueType>>
using Matrix = Tensor<2, ValueType, Allocator>;

template<class ValueType, class Allocator = tensors::detail::default_allocator<ValueType>>
using Cube = Tensor<3, ValueType, Allocator>;

} //end of ns BC

#endif /* TENSOR_ALIASES_H_ */
