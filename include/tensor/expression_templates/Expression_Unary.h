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

template<class Operation, class Value>
struct Unary_Expression : public Expression_Base<Unary_Expression<Operation, Value>>, public Operation {

    using return_type  = decltype(std::declval<Operation>()(std::declval<typename Value::value_type>()));
    using value_type  = std::remove_reference_t<std::decay_t<return_type>>;

    using system_tag  = typename Value::system_tag;

    static constexpr int tensor_dimension  = Value::tensor_dimension;
    static constexpr int tensor_iterator_dimension = Value::tensor_iterator_dimension;

    Value array;

    BCINLINE
    const Operation& get_operation() const {
    	return static_cast<const Operation&>(*this);
    }



	struct dx_defined {
		template<class T> BCINLINE
		static auto impl(const T& array, const Operation& op, BC::size_t index) {
			// ’(f(g(x))) == f’(g(x))g’(x)
			auto gx_and_dxgx = array.dx(index);	// [g(x), g’(x)]
			auto gx = gx_and_dxgx.first; // g(x)
			auto dx_gx =  gx_and_dxgx.second; // g’(x)

			auto f_gx = op(gx);
			auto dx_f_gx = op.dx(gx) * dx_gx;

			return BC::meta::make_pair(f_gx, dx_f_gx);
		}
	};

	struct dx_not_defined {
		template<class T> BCINLINE
		static auto impl(const T& array, const Operation& op, BC::size_t index) {
	    	auto value = array[index];
	    	auto gx = op(value);
	    	auto dx_gx = op.dx(value);
	    	return BC::meta::make_pair(gx, dx_gx);
		}
	};


    BCINLINE auto dx(BC::size_t index) const {
		using foo = std::conditional_t<expression_traits<Value>::derivative_is_defined, dx_defined, dx_not_defined>;
		return foo::impl(array, get_operation(), index);

    }



    template<class... args> BCINLINE
    Unary_Expression(Value v, const args&... args_)
    : Operation(args_...) , array(v) {}

    BCINLINE auto operator [](int index) const {
        return Operation::operator()(array[index]);
    }
    template<class... integers> BCINLINE
    auto operator ()(integers... index) const {
        return Operation::operator()(array(index...));
    }

    BCINLINE auto operator [](int index) {
        return Operation::operator()(array[index]);
    }
    template<class... integers> BCINLINE
    auto operator ()(integers... index) {
        return Operation::operator()(array(index...));
    }

    BCINLINE const auto inner_shape() const { return array.inner_shape(); }
    BCINLINE const auto block_shape() const { return array.block_shape(); }
    BCINLINE BC::size_t size() const { return array.size(); }
    BCINLINE BC::size_t rows() const { return array.rows(); }
    BCINLINE BC::size_t cols() const { return array.cols(); }
    BCINLINE BC::size_t dimension(int i) const { return array.dimension(i); }
    BCINLINE BC::size_t block_dimension(int i) const { return array.block_dimension(i); }

};


struct dx_forwarder { };


template<class Value>
struct Unary_Expression<dx_forwarder, Value>: public Expression_Base<Unary_Expression<dx_forwarder, Value>> {

    using system_tag  = typename Value::system_tag;

    static constexpr int tensor_dimension  = Value::tensor_dimension;
    static constexpr int tensor_iterator_dimension = Value::tensor_iterator_dimension;

    Value array;

    using value_type  = typename Value::value_type;


    template<class... args> BCINLINE
    Unary_Expression(Value v, const args&... args_)
    :array(v) {}

    BCINLINE auto operator [](int index) const {
        return array.dx(index).second;
    }
    template<class... integers> BCINLINE
    auto operator ()(integers... index) const {
        return array.dx(index...).second;
    }

    BCINLINE auto operator [](int index) {
        return array.dx(index).second;
    }
    template<class... integers> BCINLINE
    auto operator ()(integers... index) {
        return array.dx(index...).second;
    }

    BCINLINE const auto inner_shape() const { return array.inner_shape(); }
    BCINLINE const auto block_shape() const { return array.block_shape(); }
    BCINLINE BC::size_t size() const { return array.size(); }
    BCINLINE BC::size_t rows() const { return array.rows(); }
    BCINLINE BC::size_t cols() const { return array.cols(); }
    BCINLINE BC::size_t dimension(int i) const { return array.dimension(i); }
    BCINLINE BC::size_t block_dimension(int i) const { return array.block_dimension(i); }

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
