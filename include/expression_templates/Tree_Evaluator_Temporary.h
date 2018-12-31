/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef PTEE_TEMPORARY_H_
#define PTEE_TEMPORARY_H_

#include "Tree_Evaluator_Default.h"

namespace BC {
namespace et {
template<int,class,class,class...> struct ArrayExpression;
namespace tree {

template<int x, class Scalar, class Allocator>
struct evaluator<
	ArrayExpression<x, Scalar, Allocator, BC_Temporary>,
	std::enable_if_t<BC::is_temporary<ArrayExpression<x, Scalar, Allocator, BC_Temporary>>()>>
 : evaluator_default<ArrayExpression<x, Scalar, Allocator, BC_Temporary>> {

    static void deallocate_temporaries(ArrayExpression<x, Scalar, Allocator, BC_Temporary> tmp) {
        tmp.deallocate();
    }
};


}
}
}



#endif /* PTEE_TEMPORARY_H_ */
