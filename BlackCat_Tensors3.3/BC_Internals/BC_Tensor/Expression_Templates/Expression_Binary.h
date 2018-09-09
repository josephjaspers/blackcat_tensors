/*
 * BC_Expression_Binary.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */
#ifndef EXPRESSION_BINARY_POINTWISE_SAME_H_
#define EXPRESSION_BINARY_POINTWISE_SAME_H_

#include "Expression_Base.h"

namespace BC {
namespace internal {
template<class lv, class rv, class operation>
struct binary_expression : public expression_base<binary_expression<lv, rv, operation>>, public operation {

	using scalar_t = decltype(std::declval<operation>()(std::declval<typename lv::scalar_t&>(), std::declval<typename lv::scalar_t&>()));
	using mathlib_t = typename lv::mathlib_t;

	lv left;
	rv right;

	template<class L, class R> __BCinline__ const auto oper(const L& l, const R& r) const { return static_cast<const operation&>(*this)(l,r); }
	template<class L, class R> __BCinline__ 	  auto oper(const L& l, const R& r) 	   { return static_cast<      operation&>(*this)(l,r); }

	__BCinline__ static constexpr int DIMS() { return MTF::max(lv::DIMS(),rv::DIMS());}
	__BCinline__ static constexpr int ITERATOR() {
		//if dimension mismatch choose the max dimension as iterator, else choose the max iterator
		return lv::DIMS() != rv::DIMS() ? DIMS() : MTF::max(lv::ITERATOR(), rv::ITERATOR());
	}

	template<class... args>
	__BC_host_inline__ binary_expression(lv l, rv r, const args&... args_) :  operation(args_...), left(l), right(r) {}

	__BCinline__  auto  operator [](int index) const { return oper(left[index], right[index]); }
	template<class... integers> __BCinline__  auto  operator ()(integers... ints) const { return oper(left(ints...), right(ints...)); }

	__BCinline__ const auto& shape() const { return dominant_type<lv, rv>::shape(left, right); }
	__BCinline__ int size() const { return shape().size(); }
	__BCinline__ int rows() const { return shape().rows(); }
	__BCinline__ int cols() const { return shape().cols(); }
	__BCinline__ int dimension(int i) const { return shape().dimension(i); }
	__BCinline__ int block_dimension(int i) const { return shape().block_dimension(i); }
	__BCinline__ const auto inner_shape() const { return shape().inner_shape(); }
	__BCinline__ const auto block_shape() const { return shape().block_shape(); }



	__BCinline__ const auto slice(int i) const {
		using slice_lv = decltype(left.slice(i));
		using slice_rv = decltype(left.slice(i));

		return binary_expression<slice_lv, slice_rv, operation>(left.slice(i), right.slice(i),  static_cast<const operation&>(*this));
	}
	__BCinline__ const auto scalar(int i) const {
		using scalar_lv = decltype(left.scalar(i));
		using scalar_rv = decltype(left.scalar(i));

		return binary_expression<scalar_lv, scalar_rv, operation>(left.scalar(i), right.scalar(i),  static_cast<const operation&>(*this));
	}
	__BCinline__ const auto col(int i) const {
		static_assert(DIMS() == 2, "COLUMN ACCESS ONLY AVAILABLE TO MATRICES");
		return slice(i);
	}
};
}
}


#endif /* EXPRESSION_BINARY_POINTWISE_SAME_H_ */

