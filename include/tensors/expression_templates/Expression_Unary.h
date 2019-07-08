/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_EXPRESSION_UNARY_H_
#define BC_EXPRESSION_TEMPLATES_EXPRESSION_UNARY_H_


#include "Expression_Template_Base.h"
#include "Shape.h"


namespace BC {
namespace tensors {
namespace exprs { 

template<class Operation, class ArrayType>
struct Unary_Expression : public Expression_Base<Unary_Expression<Operation, ArrayType>>, public Operation {

	//TODO fix value_type deduction
//    using return_type = decltype(std::declval<Operation>()(std::declval<typename ArrayType::value_type>()));
    using value_type  = typename ArrayType::value_type;//std::remove_reference_t<std::decay_t<return_type>>;
    using dx_is_defined = std::true_type;
    using system_tag  = typename ArrayType::system_tag;

    static constexpr int tensor_dimension  = ArrayType::tensor_dimension;
    static constexpr int tensor_iterator_dimension = ArrayType::tensor_iterator_dimension;

    ArrayType array;

    BCINLINE
    const Operation& get_operation() const {
    	return static_cast<const Operation&>(*this);
    }

    template<class... args> BCINLINE
    Unary_Expression(ArrayType v, const args&... args_)
    : Operation(args_...) , array(v) {}


    // --------------- Not index Aware function (Basic)---------------- //
    template<class... integers, class Op=Operation>
    BCINLINE std::enable_if_t<!BC::oper::operation_traits<Op>::is_index_aware_function,
    value_type> operator ()(integers... index) const {
        return Operation::operator()(array(index...));
    }
    template<class Op=Operation> BCINLINE
	std::enable_if_t<!BC::oper::operation_traits<Op>::is_index_aware_function,
	value_type> operator [](int index) const {
		return Operation::operator()(array[index]);
	}

    // --------------- Index Aware function (Advanced)---------------- //
    template<class... integers, class Op=Operation> BCINLINE
    std::enable_if_t<BC::oper::operation_traits<Op>::is_index_aware_function,
    value_type> operator ()(integers... index) const {
        return Operation::operator()(array, index...);
    }

    template<class Op=Operation> BCINLINE
    std::enable_if_t<BC::oper::operation_traits<Op>::is_index_aware_function,
    value_type> operator [](int index) const {
        return Operation::operator()(array, index);
    }
    //-------- forward to const versions

    template<class... integers> BCINLINE
    value_type operator ()(integers... index) {
    	using namespace BC::meta;
        return auto_remove_const(auto_apply_const(*this)(index...));
    }

    BCINLINE value_type operator [](int index) {
    	using namespace BC::meta;
        return auto_remove_const(auto_apply_const(*this)[index]);
    }


    template<class... Indicies>
    BCINLINE auto dx(Indicies... indicies) const {
		auto gx_and_dxgx = expression_traits<ArrayType>::select_on_dx(array, indicies...);	// [g(x), g’(x)]
		auto f_gx = Operation::operator()(gx_and_dxgx.first);
		auto dx_f_gx = Operation::dx(gx_and_dxgx.first) * gx_and_dxgx.second;
		return BC::meta::make_pair(f_gx, dx_f_gx);
    }

    BCINLINE const auto inner_shape() const { return array.inner_shape(); }
    BCINLINE const auto block_shape() const { return array.block_shape(); }
    BCINLINE BC::size_t size() const { return array.size(); }
    BCINLINE BC::size_t rows() const { return array.rows(); }
    BCINLINE BC::size_t cols() const { return array.cols(); }
    BCINLINE BC::size_t dimension(int i) const { return array.dimension(i); }
    BCINLINE BC::size_t block_dimension(int i) const { return array.block_dimension(i); }
};

struct dx_forwarder {

	using index_aware_function = std::true_type;

	template<class Expression> BCINLINE
	auto operator () (const Expression& expression, size_t index) const {
		return expression.dx(index).second;
	}
	template<class Expression> BCINLINE
	auto operator () (const Expression& expression, size_t index) {
		return expression.dx(index).second;
	}

	template<class Expression, class... size_ts> BCINLINE
	auto operator () (const Expression& expression, size_ts... indicies) const {
		return expression.dx(indicies...).second;
	}
	template<class Expression, class... size_ts> BCINLINE
	auto operator () (const Expression& expression, size_ts... indicies) {
		return expression.dx(indicies...).second;
	}
};

template<class op, class expr> BCHOT
auto make_un_expr(expr e, op oper =op()) {
	static_assert(std::is_trivially_copyable<expr>::value,
			"Unary_Expressions - Arg must be trivially copyable");

	return Unary_Expression<op, expr>(e, oper);
}



} //ns BC
} //ns exprs
} //ns tensors


#endif /* EXPRESSION_UNARY_POINTWISE_CU_ */
