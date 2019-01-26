///*  Project: BlackCat_Tensors
// *  Author: JosephJaspers
// *  Copyright 2018
// *
// * This Source Code Form is subject to the terms of the Mozilla Public
// * License, v. 2.0. If a copy of the MPL was not distributed with this
// * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
//
//#ifndef BC_EXPRESSION_TEMPLATES_TENSOR_SCALAR_H_
//#define BC_EXPRESSION_TEMPLATES_TENSOR_SCALAR_H_
//
//#include "Array_Base.h"
//
//
//namespace BC {
//namespace et {
//
//
//template<class Parent>
//struct Array_Scalar : Array_Base<Array_Scalar<Parent>, 0>, Shape<0> {
//
//    using value_type = typename Parent::value_type;
//    using allocator_t = typename Parent::allocator_t;
//    using system_tag = typename Parent::system_tag;
//
//    static constexpr int ITERATOR = 0;
//    static constexpr int DIMS = 0;
//
//    value_type* array;
//
//    BCINLINE Array_Scalar(Parent parent_, BC::size_t index)
//    : array(&(parent_.memptr()[index])) {}
//
//    BCINLINE const auto& operator [] (int index) const { return array[0]; }
//    BCINLINE       auto& operator [] (int index)       { return array[0]; }
//
//    template<class... integers> BCINLINE
//    auto& operator ()(integers ... ints) {
//        return array[0];
//    }
//    template<class... integers> BCINLINE
//    const auto& operator ()(integers ... ints) const {
//        return array[0];
//    }
//
//    BCINLINE const value_type* memptr() const { return array; }
//    BCINLINE       value_type* memptr()       { return array; }
//};
//
//
//template<class internal_t>
//auto make_scalar(internal_t internal, BC::size_t  i) {
//    return Array_Scalar<internal_t>(internal, i);
//}
//
//
//}
//}
//
//
//
//#endif /* TENSOR_SLICE_CU_ */