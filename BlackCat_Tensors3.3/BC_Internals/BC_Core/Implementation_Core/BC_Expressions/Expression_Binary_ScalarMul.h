/*
 * Expression_Binary_ScalarMul.h
 *
 *  Created on: Mar 20, 2018
 *      Author: joseph
 */

#ifndef EXPRESSION_BINARY_SCALARMUL_H_
#define EXPRESSION_BINARY_SCALARMUL_H_

#include "Expression_Base.h"
#include "Expression_Binary_Functors.h"
#include <type_traits>
namespace BC {

template<class lv, class rv, class left = void>
struct dominant_type {
	__BCinline__ static const auto& shape(const lv& l, const rv& r) {
		return l;
	}
};
template<class lv, class rv>
struct dominant_type<lv, rv, std::enable_if_t<(lv::RANK < rv::RANK)>> {

	__BCinline__ static const auto& shape(const lv& l, const rv& r) {
		return r;
	}
};

template<class lv, class rv, class left = void>
struct inferior_type {
	__BCinline__ static const auto& shape(const lv& l, const rv& r) {
		return l;
	}
};
template<class lv, class rv>
struct inferior_type<lv, rv, std::enable_if_t<(lv::RANK > rv::RANK)>> {

	__BCinline__ static const auto& shape(const lv& l, const rv& r) {
		return r;
	}
};


template<class T, class lv, class rv>
struct binary_expression_scalarMul : public expression<T, binary_expression_scalarMul<T, operation, lv, rv>> {

	mul oper;

	lv left;
	rv right;

	auto getScalar() {
		return inferior_type<lv, rv>::shape(left,right);
	}
	auto getArray() {
		return dominant_type<lv, rv>::shape(left, right);
	}

	static constexpr int RANK() { return lv::RANK() > rv::RANK() ? lv::RANK() : rv::RANK();}
	static constexpr bool lv_dom = (lv::RANK() > rv::RANK());

	__BCinline__ const auto& shape() const {
		return dominant_type<lv, rv>::shape(left, right);
	}

	template<class L, class R>
	__BCinline__  binary_expression_scalarMul(L l, R r) :
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

	void printDimensions() 		const { shape().printDimensions();   }
	void printLDDimensions()	const { shape().printLDDimensions(); }


};


}


#endif /* EXPRESSION_BINARY_SCALARMUL_H_ */
