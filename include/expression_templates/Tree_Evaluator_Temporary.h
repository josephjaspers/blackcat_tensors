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
namespace tree {

template<int x, class Scalar, class Allocator>
struct evaluator<
	ArrayExpression<x, Scalar, Allocator, Temporary<x, Scalar, Allocator>>,
	std::enable_if_t<is_array<ArrayExpression<x, Scalar, Allocator, Temporary<x, Scalar, Allocator>>>()>>
 : evaluator_default<ArrayExpression<x, Scalar, Allocator, Temporary<x, Scalar, Allocator>>> {

    static void deallocate_temporaries(ArrayExpression<x, Scalar, Allocator, Temporary<x, Scalar, Allocator>> tmp) {
        tmp.deallocate();
    }
};


}
}
}



#endif /* PTEE_TEMPORARY_H_ */
