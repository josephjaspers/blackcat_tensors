/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_FUNCTION_TRANSPOSE_H_
#define BC_EXPRESSION_TEMPLATES_FUNCTION_TRANSPOSE_H_
#include <vector>
#include "Expression_Base.h"


namespace BC {
namespace exprs {


template<class Value, class System_Tag>
struct Unary_Expression<Value, oper::transpose<System_Tag>>
    : Expression_Base<Unary_Expression<Value, oper::transpose<System_Tag>>> {

    using value_type  = typename Value::value_type;
    using system_tag = System_Tag;

    static constexpr int DIMS = Value::DIMS;
    static constexpr int ITERATOR = DIMS > 1? DIMS :0;

    Value array;

    Unary_Expression(Value p) : array(p) {}

    BCINLINE const auto inner_shape() const {
        return make_lambda_array<DIMS>([=](int i) {
            if (DIMS >= 2)
                return i == 0 ? array.cols() : i == 1 ? array.rows() : array.dimension(i);
            else if (DIMS == 2)
                return i == 0 ? array.cols() : i == 1 ? array.rows() : 1;
            else if (DIMS == 1)
                return i == 0 ? array.rows() : 1;
            else
                return 1;
        });
    }

    BCINLINE
    const auto block_shape() const {
        return make_lambda_array<DIMS>([=](int i) {
            return i == 0 ? array.cols() : 1 == 1 ? array.rows() : array.block_dimension(i);
        });
    }

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

    BCINLINE
    BC::size_t  block_dimension(int i) const {
    	return block_shape()[i];
    }

    template<class... ints> BCINLINE
    auto operator ()(BC::size_t m, BC::size_t n, ints... integers) const -> decltype(array(n,m)) {
        return array(n,m, integers...);
    }

};

template<class expr_t>
auto make_transpose(expr_t expr) {
	using internal_t = std::decay_t<decltype(expr.internal())>;
	using system_tag = typename internal_t::system_tag;
	return Unary_Expression<internal_t, oper::transpose<system_tag>>(expr.internal());
}

}
}


#endif /* EXPRESSION_UNARY_MATRIXTRANSPOSITION_H_ */
