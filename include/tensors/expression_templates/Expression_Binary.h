/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_EXPRESSION_BINARY_H_
#define BC_EXPRESSION_TEMPLATES_EXPRESSION_BINARY_H_

#include "Expression_Template_Base.h"
#include "Shape.h"

namespace BC {
namespace tensors {
namespace exprs {

template<class Operation, class Lv, class Rv>
struct Binary_Expression : public Expression_Base<Binary_Expression<Operation, Lv, Rv>>, public Operation {

	using lv_value_t = typename Lv::value_type;
	using rv_value_t = typename Rv::value_type;
    using return_type = decltype(std::declval<Operation>()(std::declval<lv_value_t&>(), std::declval<rv_value_t&>()));
    using value_type  = std::remove_reference_t<std::decay_t<return_type>>;
    using system_tag  = typename Lv::system_tag;

    static constexpr int tensor_dimension = BC::traits::max(Lv::tensor_dimension, Rv::tensor_dimension);
    static constexpr int tensor_iterator_dimension =
    		Lv::tensor_dimension != Rv::tensor_dimension ?
    		tensor_dimension :
    		BC::traits::max(
    				Lv::tensor_iterator_dimension,
    				Rv::tensor_iterator_dimension,
    				Lv::tensor_dimension,
    				Rv::tensor_dimension);
    Lv left;
    Rv right;

    Operation get_operation() const {
    	return static_cast<const Operation&>(*this);
    }

    template<class... args> BCHOT
    Binary_Expression(Lv l, Rv r, const args&... args_):
    Operation(args_...), left(l), right(r) {}


	BCINLINE auto  operator [](int index) const {
    	return Operation::operator()(left[index], right[index]);
    }

    template<class... integers, class=std::enable_if_t<(sizeof...(integers)>=tensor_iterator_dimension)>>
    BCINLINE auto  operator ()(integers... ints) const {
    	return Operation::operator()(left(ints...), right(ints...));
    }

    BCINLINE
    auto  operator [](int index) {
    	return Operation::operator()(left[index], right[index]);
    }

    template<class... integers, class=std::enable_if_t<(sizeof...(integers)>=tensor_iterator_dimension)>>
    BCINLINE auto  operator ()(integers... ints) {
     	return Operation::operator()(left(ints...), right(ints...));
    }

private:
    BCINLINE const auto& shape() const {
    	struct Lv_shape {
    		static BCINLINE
    		const Lv& get_shape(const Lv& left, const Rv& right) { return left; }
    	};
    	struct Rv_shape {
    		static BCINLINE
    		const Rv& get_shape(const Lv& left, const Rv& right) { return right; }
    	};

    	using impl = std::conditional_t<
    			(Lv::tensor_dimension >= Rv::tensor_dimension), Lv_shape, Rv_shape>;
    	return impl::get_shape(left, right);
    }

public:
    BCINLINE BC::size_t size() const { return shape().size(); }
    BCINLINE BC::size_t rows() const { return shape().rows(); }
    BCINLINE BC::size_t cols() const { return shape().cols(); }
    BCINLINE BC::size_t dimension(int i) const { return shape().dimension(i); }
    BCINLINE BC::size_t block_dimension(int i) const { return shape().block_dimension(i); }
    BCINLINE const auto inner_shape() const { return shape().inner_shape(); }
};

template<class Op, class Lv, class Rv, class... Args> BCHOT
auto make_bin_expr(Lv left, Rv right, Args&&... args) {
	return Binary_Expression<Op,Lv, Rv>(left, right, args...);
}


} //ns BC
} //ns exprs
} //ns tensors


#endif /* EXPRESSION_BINARY_POINTWISE_SAME_H_ */


