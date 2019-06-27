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
namespace {
template<class T>
using default_allocator = BC::Allocator<default_system_tag_t, T>;
}
template<int dimension, class ValueType, class Allocator=default_allocator<ValueType>>
using Tensor = Tensor_Base<exprs::Array<dimension, ValueType, Allocator>>;

template<class ValueType, class Allocator = default_allocator<ValueType>> using Scalar = Tensor<0, ValueType, Allocator>;
template<class ValueType, class Allocator = default_allocator<ValueType>> using Vector = Tensor<1, ValueType, Allocator>;
template<class ValueType, class Allocator = default_allocator<ValueType>> using Matrix = Tensor<2, ValueType, Allocator>;
template<class ValueType, class Allocator = default_allocator<ValueType>> using Cube = Tensor<3, ValueType, Allocator>;

template<int dimension, class ValueType, class Allocator=default_allocator<ValueType>>
using Tensor_View = Tensor_Base<BC::exprs::Array_Const_View<dimension, ValueType, Allocator>>;

template<class ValueType, class Allocator = default_allocator<ValueType>> using Scalar_View = Tensor_View<0, ValueType, Allocator>;
template<class ValueType, class Allocator = default_allocator<ValueType>> using Vector_View = Tensor_View<1, ValueType, Allocator>;
template<class ValueType, class Allocator = default_allocator<ValueType>> using Matrix_View = Tensor_View<2, ValueType, Allocator>;
template<class ValueType, class Allocator = default_allocator<ValueType>> using Cube_View   = Tensor_View<3, ValueType, Allocator>;

template<int X, class Expression, class=std::enable_if_t<Expression::tensor_dimension == X>>
using TensorXpr = Tensor_Base<Expression>;

template<class Expression> using ScalarXpr = TensorXpr<0, Expression>;
template<class Expression> using VectorXpr = TensorXpr<1, Expression>;
template<class Expression> using MatrixXpr = TensorXpr<2, Expression>;
template<class Expression> using CubeXpr = TensorXpr<3, Expression>;

}

#endif /* TENSOR_ALIASES_H_ */
