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
namespace exprs {


template<class Lv, class Rv, class Operation>
struct Binary_Expression : public Expression_Base<Binary_Expression<Lv, Rv, Operation>>, public Operation {

	using lv_value_t = typename Lv::value_type;
	using rv_value_t = typename Rv::value_type;
    using return_type = decltype(std::declval<Operation>()(std::declval<lv_value_t&>(), std::declval<rv_value_t&>()));
    using value_type  = std::remove_reference_t<std::decay_t<return_type>>;
    using system_tag  = typename Lv::system_tag;

    static constexpr int DIMS = Lv::DIMS > Rv::DIMS ?  Lv::DIMS : Rv::DIMS;
    static constexpr int ITERATOR = Lv::DIMS != Rv::DIMS ? DIMS : BC::meta::max(Lv::ITERATOR, Rv::ITERATOR);

    Lv left;
    Rv right;

    template<class... args> BCHOT
    Binary_Expression(Lv l, Rv r, const args&... args_) :  Operation(args_...), left(l), right(r) {}


	template<class L, class R> BCINLINE
	const auto oper(const L& l, const R& r) const {
		return static_cast<const Operation&>(*this)(l,r);
	}

	template<class L, class R> BCINLINE
	auto oper(const L& l, const R& r) {
		return static_cast< Operation&>(*this)(l,r);
	}


    BCINLINE
    auto  operator [](int index) const {
    	return oper(left[index], right[index]);
    }

    template<class... integers> BCINLINE
    auto  operator ()(integers... ints) const {
    	return oper(left(ints...), right(ints...));
    }
private:
    BCINLINE const auto& shape() const { return dominant_type<Lv, Rv>::shape(left, right); }
public:
    BCINLINE BC::size_t size() const { return shape().size(); }
    BCINLINE BC::size_t rows() const { return shape().rows(); }
    BCINLINE BC::size_t cols() const { return shape().cols(); }
    BCINLINE BC::size_t dimension(int i) const { return shape().dimension(i); }
    BCINLINE BC::size_t block_dimension(int i) const { return shape().block_dimension(i); }
    BCINLINE const auto inner_shape() const { return shape().inner_shape(); }
    BCINLINE const auto block_shape() const { return shape().block_shape(); }
};

template<class Op, class Lv, class Rv, class... Args> BCHOT
auto make_bin_expr(Lv left, Rv right, const Args&... args) {
	return Binary_Expression<
			std::decay_t<decltype(left.internal())>,
			std::decay_t<decltype(right.internal())>,
			Op>(left.internal(), right.internal(), args...);
}
template<class Lv, class Rv, class Op> BCHOT
auto make_bin_expr(Lv left, Rv right, Op oper) {
	return Binary_Expression<
			std::decay_t<decltype(left.internal())>,
			std::decay_t<decltype(right.internal())>,
			Op>(left.internal(), right.internal(), oper);
}


}
}


#endif /* EXPRESSION_BINARY_POINTWISE_SAME_H_ */


