/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_FUNCTION_TRANSPOSE_H_
#define BC_EXPRESSION_TEMPLATES_FUNCTION_TRANSPOSE_H_

#include "expression_template_base.h"

namespace bc {
namespace tensors {
namespace exprs { 


template<class Value, class System_Tag>
struct Un_Op<oper::transpose<System_Tag>, Value>:
		Expression_Base<Un_Op<oper::transpose<System_Tag>, Value>>,
		oper::transpose<System_Tag> {

	using value_type  = typename Value::value_type;
	using system_tag = System_Tag;

	static constexpr int tensor_dim = Value::tensor_dim;
	static constexpr int tensor_iterator_dim =
			tensor_dim > 1? tensor_dim :0;

	Value array;


	Un_Op(
			Value array,
			oper::transpose<System_Tag> = oper::transpose<System_Tag>()):
		array(array) {}

	static oper::transpose<System_Tag> get_operation() {
		return oper::transpose<System_Tag>();
	}

	template<class ADL_Integer> BCINLINE
	auto operator [] (ADL_Integer i) const -> decltype(array[i]) {
		return array[i];
	}

	BCINLINE bc::size_t  size() const { return array.size(); }
	BCINLINE bc::size_t  rows() const { return array.cols(); }
	BCINLINE bc::size_t  cols() const { return array.rows(); }

	BCINLINE bc::size_t  dim(int i) const {
		if (i == 0)
			return array.cols();
		else if (i == 1)
			return array.rows();
		else
			return array.dim(i);
	}

	template<class... ints> BCINLINE
	auto operator ()(bc::size_t m, bc::size_t n, ints... integers) const
		-> decltype(array(n,m, integers...)) {
		return array(n,m, integers...);
	}
	template<class... ints> BCINLINE
	auto operator ()(bc::size_t m, bc::size_t n, ints... integers)
		-> decltype(array(n,m, integers...)) {
		return array(n,m, integers...);
	}
};


template<class expr_t>
auto make_transpose(expr_t expr) {
	using expression_template_t = std::decay_t<decltype(expr.expression_template())>;
	using system_tag = typename expression_template_t::system_tag;
	return Un_Op<oper::transpose<system_tag>, expression_template_t>(expr.expression_template());
}

template<class Array, class SystemTag>
auto make_transpose(Un_Op<oper::transpose<SystemTag>, Array> expression) {
	return expression.array;
}

} //ns BC
} //ns exprs
} //ns tensors

#endif /* EXPRESSION_UNARY_MATRIXTRANSPOSITION_H_ */
