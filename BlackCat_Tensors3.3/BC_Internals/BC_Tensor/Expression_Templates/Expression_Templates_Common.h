/*
 * Expression_Templates_Common.h
 *
 *  Created on: Sep 9, 2018
 *      Author: joseph
 */

#ifndef BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_EXPRESSION_TEMPLATES_COMMON_H_
#define BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_EXPRESSION_TEMPLATES_COMMON_H_

#include "Utility/MetaTemplateFunctions.h"
#include "Utility/ShapeHierarchy.h"
#include "Utility/Utility_Structs.h"

namespace BC {
namespace internal {

template<class T> using i_mathlib_of = typename T::mathlib_t;
template<class T> using i_scalar_of  = typename T::scalar_t;

}
}


#endif /* BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_EXPRESSION_TEMPLATES_COMMON_H_ */
