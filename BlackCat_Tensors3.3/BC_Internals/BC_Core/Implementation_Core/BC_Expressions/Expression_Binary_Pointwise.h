/*
 * BC_Expression_Binary.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */
#ifndef EXPRESSION_BINARY_POINTWISE_SAME_H_
#define EXPRESSION_BINARY_POINTWISE_SAME_H_

#include "Expression_Base.h"
#include "Expression_Binary_Functors.h"
#include <type_traits>
namespace BC {

template<class T, class operation, class lv, class rv>
struct binary_expression : public expression<T, binary_expression<T, operation, lv, rv>> {

	operation oper;

	lv left;
	rv right;

	static constexpr int RANK() { return lv::RANK() > rv::RANK() ? lv::RANK() : rv::RANK();}
	static constexpr bool lv_dom = (lv::RANK() > rv::RANK());

	__BCinline__ const auto& shape() const {
		return dominant_type<lv, rv>::shape(left, right);
	}

	template<class L, class R>
	__BCinline__  binary_expression(L l, R r) :
			left(l), right(r) {
	}

	__BCinline__  auto operator [](int index) const -> decltype(oper(left[index], right[index])) {
		return oper(left[index], right[index]);
	}

	__BCinline__ int rank() const { return shape().rank(); }
	__BCinline__ int rows() const { return shape().rows(); };
	__BCinline__ int cols() const { return shape().cols(); };
	__BCinline__ int size() const { return shape().size(); };
	__BCinline__ int LD_rows() const { return shape().LD_rows(); }
	__BCinline__ int LD_cols() const { return shape().LD_cols(); }
	__BCinline__ int dimension(int i)		const { return shape().dimension(i); }
	__BCinline__ const auto innerShape() const { return shape().innerShape(); }
	__BCinline__ const auto outerShape() const { return shape().outerShape(); }


	template<class v, class alt>
	using expr_type = std::conditional_t<v::RANK() == 0, v, alt>;

	__BCinline__ const auto slice(int i) const {
		return binary_expression<T, operation, decltype(left.slice(0)), decltype(right.slice(0))>(left.slice(i), right.slice(i));
	}
	__BCinline__ const auto row(int i) const {
		return binary_expression<T, operation, expr_type<lv, decltype(left.row(0))>, expr_type<rv, decltype(right.row(0))>>(left.row(i), right.row(i)); }

	__BCinline__ const auto col(int i) const {
		return binary_expression<T, operation, expr_type<lv, decltype(left.col(0))>, expr_type<rv, decltype(right.col(0))>>(left.col(i), right.col(i)); }


	void printDimensions() 		const { shape().printDimensions();   }
	void printLDDimensions()	const { shape().printLDDimensions(); }
};


//class specifically for matrix multiplication
template<class T, class lv, class rv>
struct binary_expression_scalar_mul : expression<T, binary_expression_scalar_mul<T, lv, rv>> {

	mul oper;

	lv left;
	rv right;

	static constexpr int RANK() { return lv::RANK() > rv::RANK() ? lv::RANK() : rv::RANK();}
	static constexpr bool lv_dom = (lv::RANK() > rv::RANK());

	__BCinline__ const auto& shape() const {
		return dominant_type<lv, rv>::shape(left, right);
	}

	__BCinline__  binary_expression_scalar_mul(lv l, rv r) :
			left(l), right(r) {
	}

	__BCinline__  auto operator [](int index) const -> decltype(oper(left[index], right[index])) {
		return oper(left[index], right[index]);
	}

	__BCinline__  int rank() const { return shape().rank(); }
	__BCinline__  int rows() const { return shape().rows(); };
	__BCinline__  int cols() const { return shape().cols(); };
	__BCinline__  int size() const { return shape().size(); };
	__BCinline__  int LD_rows() const { return shape().LD_rows(); }
	__BCinline__ int LD_cols() const { return shape().LD_cols(); }
	__BCinline__ int dimension(int i)		const { return shape().dimension(i); }
	__BCinline__ const auto innerShape() const { return shape().innerShape(); }
	__BCinline__ const auto outerShape() const { return shape().outerShape(); }


	template<class v, class alt>
	using expr_type = std::conditional_t<v::RANK() == 0, v, alt>;

	__BCinline__ const auto slice(int i) const {
		return binary_expression<T, mul, decltype(left.slice(0)), decltype(right.slice(0))>(left.slice(i), right.slice(i));
	}
	__BCinline__ const auto row(int i) const {
		return binary_expression<T, mul, expr_type<lv, decltype(left.row(0))>, expr_type<rv, decltype(right.row(0))>>(left.row(i), right.row(i)); }

	__BCinline__ const auto col(int i) const {
		return binary_expression<T, mul, expr_type<lv, decltype(left.col(0))>, expr_type<rv, decltype(right.col(0))>>(left.col(i), right.col(i)); }


	void printDimensions() 		const { shape().printDimensions();   }
	void printLDDimensions()	const { shape().printLDDimensions(); }
};

}

#endif /* EXPRESSION_BINARY_POINTWISE_SAME_H_ */

