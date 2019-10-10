/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BLACKCAT_BLACKCAT_COMMON_H_
#define BLACKCAT_BLACKCAT_COMMON_H_

#include <type_traits>

#include "expression_templates/Expression_Templates.h"
#include "iterators/Iterators.h"

namespace BC {
namespace tensors {

template<class> class Tensor_Base;

template<class ExpressionTemplate>
auto make_tensor(ExpressionTemplate expression) {
	static_assert(
		exprs::expression_traits<ExpressionTemplate>::is_expression_template::value,
		"Make Tensor can only be used with Expression_Template");
	return Tensor_Base<ExpressionTemplate>(expression);
}
}

}

#endif /* BC_INTERNALS_BC_TENSOR_TENSOR_COMMON_H_ */
