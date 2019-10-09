/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_FUNCTION_TRANSPOSE_H_
#define BC_EXPRESSION_TEMPLATES_FUNCTION_TRANSPOSE_H_

#include "Expression_Template_Base.h"

namespace BC {
namespace tensors {
namespace exprs { 


template<class Value, class System_Tag>
struct Unary_Expression<oper::transpose<System_Tag>, Value>:
		Expression_Base<Unary_Expression<oper::transpose<System_Tag>, Value>>,
		oper::transpose<System_Tag> {

	using value_type  = typename Value::value_type;
	using system_tag = System_Tag;

	static constexpr int tensor_dimension = Value::tensor_dimension;
	static constexpr int tensor_iterator_dimension =
			tensor_dimension > 1? tensor_dimension :0;

	Value array;


	Unary_Expression(
			Value array,
			oper::transpose<System_Tag> = oper::transpose<System_Tag>()):
		array(array) {}

	BCINLINE
	auto operator [] (int i) const -> decltype(array[0]) {
		return array[i];
	}

	BCINLINE BC::size_t  size() const { return array.size(); }
	BCINLINE BC::size_t  rows() const { return array.cols(); }
	BCINLINE BC::size_t  cols() const { return array.rows(); }

	BCINLINE BC::size_t  dimension(int i) const {
		if (i == 0)
			return array.cols();
		else if (i == 1)
			return array.rows();
		else
			return array.dimension(i);
	}

	template<class... ints> BCINLINE
	auto operator ()(BC::size_t m, BC::size_t n, ints... integers) const
		-> decltype(array(n,m)) {
		return array(n,m, integers...);
	}
	template<class... ints> BCINLINE
	auto operator ()(BC::size_t m, BC::size_t n, ints... integers)
		-> decltype(array(n,m)) {
		return array(n,m, integers...);
	}
};


template<class expr_t>
auto make_transpose(expr_t expr) {
	using internal_t = std::decay_t<decltype(expr.internal())>;
	using system_tag = typename internal_t::system_tag;
	return Unary_Expression<oper::transpose<system_tag>, internal_t>(expr.internal());
}

template<class Array, class SystemTag>
auto make_transpose(Unary_Expression<oper::transpose<SystemTag>, Array> expression) {
	return expression.array;
}

} //ns BC
} //ns exprs
} //ns tensors

#endif /* EXPRESSION_UNARY_MATRIXTRANSPOSITION_H_ */
