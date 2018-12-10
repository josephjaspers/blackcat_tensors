/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_TREE_EVALUATOR_COMMON_H_
#define BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_TREE_EVALUATOR_COMMON_H_


#include "utility/MetaTemplateFunctions.h"
#include "Internal_Common.h"
#include <type_traits>

#include "Tree_Struct_Temporary.h"
#include "Tree_Struct_Injector.h"

#include "operations/Unary.h"
#include "operations/Binary.h"
#include "operations/BLAS.h"

#include "Tree_Functions.h"

#ifdef BC_TREE_OPTIMIZER_DEBUG
#define BC_TREE_OPTIMIZER_STDOUT(literal) std::cout << literal << std::endl;
#else
#define BC_TREE_OPTIMIZER_STDOUT(literal)
#endif

namespace BC {
namespace et {
namespace tree {

//initial forward decl
template<class T, class=void>
struct evaluator;

}
}
}



#endif /* BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_TREE_EVALUATOR_COMMON_H_ */
