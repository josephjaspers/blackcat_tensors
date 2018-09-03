/*
 * BC_Expression_Base.h
 *
 *  Created on: Dec 11, 2017
 *      Author: joseph
 */

#ifndef EXPRESSION_BASE_H_
#define EXPRESSION_BASE_H_

#include "BC_Internal_Base.h"
#include "Shape_Expression.h"
namespace BC {
namespace internal {

template<class derived>
struct expression_base
		: BC_internal_base<derived>, Shape_Expression<derived> {
};
}
}

#endif /* EXPRESSION_BASE_H_ */
