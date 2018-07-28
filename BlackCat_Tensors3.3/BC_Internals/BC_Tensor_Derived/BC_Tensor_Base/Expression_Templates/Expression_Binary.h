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

	__BCinline__ static constexpr int DIMS() { return MTF::max(lv::DIMS(),rv::DIMS());}
	__BCinline__ static constexpr int ITERATOR() {
		//if dimension mismatch choose the max dimension as iterator, else choose the max iterator
		return lv::DIMS() != rv::DIMS() ? DIMS() : MTF::max(lv::ITERATOR(), rv::ITERATOR());
	}

	__BCinline__  binary_expression(lv l, rv r, operation oper_ = operation()) : left(l), right(r), oper(oper_) {}

	__BCinline__  auto  operator [](int index) const { return oper(left[index], right[index]); }
	template<class... integers> __BCinline__  auto  operator ()(integers... ints) const { return oper(left(ints...), right(ints...)); }

	__BCinline__ const auto& shape() const { return dominant_type<lv, rv>::shape(left, right); }
	__BCinline__ int size() const { return shape().size(); }
	__BCinline__ int rows() const { return shape().rows(); }
	__BCinline__ int cols() const { return shape().cols(); }
	__BCinline__ int dimension(int i) const { return shape().dimension(i); }
	__BCinline__ int outer_dimension() const { return shape().outer_dimension(); }
	__BCinline__ const auto inner_shape() const { return shape().inner_shape(); }
	__BCinline__ const auto slice(int i) const {
		using slice_lv = decltype(left.slice(i));
		using slice_rv = decltype(left.slice(i));

		return binary_expression<slice_lv, slice_rv, operation>(left.slice(i), right.slice(i), oper);
	}
	__BCinline__ const auto scalar(int i) const {
		using scalar_lv = decltype(left.scalar(i));
		using scalar_rv = decltype(left.scalar(i));

		return binary_expression<scalar_lv, scalar_rv, operation>(left.scalar(i), right.scalar(i), oper);
	}

	__BCinline__ const auto col(int i) const {
		static_assert(DIMS() == 2, "COLUMN ACCESS ONLY AVAILABLE TO MATRICES");
		return slice(i);
	}
	__BCinline__ const auto row(int i) const {
		static_assert(DIMS() == 2 || DIMS() == 1, "ROW ACCESS ONLY AVAILABLE TO MATRICES");
		return slice(i);
	}
};
}
}


#endif /* EXPRESSION_BINARY_POINTWISE_SAME_H_ */

