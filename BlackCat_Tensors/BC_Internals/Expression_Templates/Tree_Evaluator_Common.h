/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_TREE_EVALUATOR_COMMON_H_
#define BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_TREE_EVALUATOR_COMMON_H_


#include "Utility/MetaTemplateFunctions.h"
#include <type_traits>

#include "Tree_Struct_Temporary.h"
#include "Tree_Struct_Injector.h"

#include "Operations/Unary.h"
#include "Operations/Binary.h"
#include "Operations/BLAS.h"

#include "Tree_Functions.h"



namespace BC {
namespace et     {
namespace tree {


//initial forward decl
template<class T, class enabler = void>
struct evaluator;

}
}
}



#endif /* BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_TREE_EVALUATOR_COMMON_H_ */
