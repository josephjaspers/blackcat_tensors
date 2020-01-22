/*  Project: BlackCat_Scalars
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_FUNCTION_SUM_H_
#define BC_EXPRESSION_TEMPLATES_FUNCTION_SUM_H_

#include "expression_template_base.h"
#include "tree_evaluator.h"

#include "functions/reductions/reductions.h"


namespace bc {
namespace tensors {
namespace exprs { 

template<class SystemTag>
struct Sum {};

template<class ArrayType, class SystemTag>
struct Un_Op<Sum<SystemTag>, ArrayType>:
	Expression_Base<Un_Op<Sum<SystemTag>, ArrayType>>,
	Shape<0>,
	Sum<SystemTag> {

	using value_type = typename ArrayType::value_type;
	using system_tag = SystemTag;
	using requires_greedy_evaluation = std::true_type;

	static constexpr int tensor_dim  = 0;
	static constexpr int tensor_iterator_dim = 0;

	ArrayType array;

	using Shape<0>::inner_shape;

	Un_Op(ArrayType array, Sum<system_tag> = Sum<system_tag>()):
		array(array) {}

	static Sum<SystemTag> get_operation() {
		return Sum<SystemTag>();
	}

	template<class Scalar, int Alpha, int Beta, class Stream>
	void eval(Output_Data<Scalar, Alpha, Beta> output, Stream stream) const {
		static_assert(Scalar::tensor_dim==0, "Output must be a scalar");

		//TODO handle alpha/beta scalars
		bc::tensors::exprs::functions::Reduce<system_tag>::sum(
				stream,
				output.data(),
				array);
	}
};


} //ns BC
} //ns exprs
} //ns tensors



#endif /* FUNCTION_DOT_H_ */
