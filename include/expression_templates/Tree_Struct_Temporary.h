/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef TEMPORARY_ARRAY_H_
#define TEMPORARY_ARRAY_H_

namespace BC {
namespace et {
template<int,class,class> struct Temporary;
template<int,class,class, class> struct ArrayExpression;

namespace tree {

/*
 *Tag for BC::Array to flag for deletion after using the tree-evaluator
 *
 */

//template<class data_t>
//struct temporary : data_t {
//    using data_t::data_t;
//};

template<class T>
struct is_temporary : std::false_type {};

template<int dims, class Scalar, class Allocator>
struct is_temporary<ArrayExpression<dims, Scalar, Allocator, Temporary<dims, Scalar, Allocator>>> : std::true_type {};


}
}
}



#endif /* TEMPORARY_ARRAY_H_ */
