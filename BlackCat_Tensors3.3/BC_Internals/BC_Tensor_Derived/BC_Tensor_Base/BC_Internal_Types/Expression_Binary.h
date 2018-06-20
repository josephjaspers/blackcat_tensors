/*
 * BC_Expression_Binary.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */
#ifndef EXPRESSION_BINARY_POINTWISE_SAME_H_
#define EXPRESSION_BINARY_POINTWISE_SAME_H_

#include "Expression_Base.h"
#include "Parse_Tree_Functions.h"


namespace BC {
namespace internal {
template<class lv, class rv, class operation>
struct binary_expression : public expression_base<binary_expression<lv, rv, operation>> {

	operation oper;

	lv left;
	rv right;

	__BCinline__ static constexpr int DIMS() { return lv::DIMS() > rv::DIMS() ? lv::DIMS() : rv::DIMS();}
	__BCinline__ static constexpr int ITERATOR() { return MTF::max(lv::ITERATOR(), rv::ITERATOR()); }
	__BCinline__ static constexpr bool INJECTABLE() { return lv::INJECTABLE() || rv::INJECTABLE(); }

	__BCinline__  binary_expression(lv l, rv r, operation oper_ = operation()) : left(l), right(r), oper(oper_) {}

	__BCinline__  auto  operator [](int index) const { return oper(left[index], right[index]); }
	template<class... integers> __BCinline__  auto  operator ()(integers... ints) const { return oper(left(ints...), right(ints...)); }

	__BCinline__ const auto& shape() const { return dominant_type<lv, rv>::shape(left, right); }
	__BCinline__ const auto  inner_shape() const { return shape().inner_shape(); }
	__BCinline__ const auto  outer_shape() const { return shape().outer_shape(); }
};
}
}


#endif /* EXPRESSION_BINARY_POINTWISE_SAME_H_ */

