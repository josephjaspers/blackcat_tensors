/*  Project: BlackCat_Tensors
 *  Author: Joseph Jaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef PTE_ARRAY_H_
#define PTE_ARRAY_H_

#include "Tree_Evaluator_Common.h"
#include "Tree_Evaluator_Default.h"
#include "Array_Base.h"

namespace BC {
namespace et     {
namespace tree {


template<class T> struct evaluator<T, std::enable_if_t<is_array<T>()>> : evaluator_default<T> {};


}
}
}



#endif /* PTE_ARRAY_H_ */
