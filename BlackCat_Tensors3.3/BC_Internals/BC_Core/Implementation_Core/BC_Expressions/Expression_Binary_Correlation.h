/*
 * Expressions_Binary_Correlation.h
 *
 *  Created on: Mar 22, 2018
 *      Author: joseph
 */

#ifndef EXPRESSIONS_BINARY_CORRELATION_H_
#define EXPRESSIONS_BINARY_CORRELATION_H_

#include "Expression_Base.h"

template<class T, class lv, class rv>
struct binary_expression_correlation {

	static constexpr bool RANK_EQUALITY = lv::RANK() == rv::RANK();
	static_assert(RANK_EQUALITY, "CORRELATION CURRENTLY ONLY SUPPORTED FOR SAME ORDER TENSORS");
	static constexpr int RANK() { return lv::RANK(); }
	template<bool val> using tf = std::conditional_t<val, std::true_type, std::false_type>;

	lv left;  //krnl
	rv right; //img

	binary_expression_correlation(lv l_, rv r_) :left(l_), right(r_) {}

	template<class K, class I>
	T axpy(int index, const K& krnl, const I& img, int order = RANK()) const {
		T sum = 0;
//		if (order == 1)
//
//			for (int i = 0; i < left.rows(); ++i)
//				sum += krnl[i] * img[index + i];
//
//		else
//			for (int i = 0; i < left.dimension(order - 1); ++i)
//				sum += axpy(index, krnl.slice(i), img.slice(i), order - 1);
//
		return sum;
	}

	__BCinline__  T operator [] (int i) const {
		return axpy(i, left, right, rank());
	}

	int size() const {
		int sz = 1;
		for (int i = 0; i < RANK() + 1; ++i)
			sz *= dimension(i);
		return sz;
	}



	__BCinline__ int rank() const { return RANK(); }
	__BCinline__ int rows() const { return right.rows() - left.rows() + 1; };
	__BCinline__ int cols() const { return right.rows() - left.rows() + 1; };

	__BCinline__ int LD_rows() const { return rows(); }
	__BCinline__ int LD_cols() const { return size(); }
	__BCinline__ int dimension(int i) const { return (right.dimension(i) - left.dimension(i) + 1); }


	__BCinline__ int last(int x) {
		return x;
	}
	template<class ... integers> __BCinline__ int last(int x, integers ... ints) {
		return last(ints...);
	}

	__BCinline__ const auto innerShape() const {
		return ref_array(*this);
	}

	__BCinline__ const auto outerShape() const {
		stack_array<int, RANK()> ary;
		ary[0] = rows();
		for (int i = 1; i < RANK(); ++i) {
			ary[i] = dimension(i) * ary[i - 1];
		}
		return ary;
	}

	template<class v, class alt>
	using expr_type = std::conditional_t<v::RANK() == 0, v, alt>;

	__BCinline__ const auto slice(int i) const {
		std::cout << " correlation of slice is not well defined " << std::endl;
		return binary_expression_correlation<T, lv, decltype(right.slice(0))>(left.slice(i), right.slice(i));
	}
	__BCinline__ const auto row(int i) const {
		std::cout << " correlation of slice is not well defined " << std::endl;
		return binary_expression_correlation<T, lv, expr_type<rv, decltype(right.row(0))>>(left.row(i), right.row(i)); }

	__BCinline__ const auto col(int i) const {
		std::cout << " correlation of slice is not well defined " << std::endl;
		return binary_expression_correlation<T, lv, expr_type<rv, decltype(right.col(0))>>(left.col(i), right.col(i)); }


	void printDimensions()const  {
		for (int i = 0; i < RANK(); ++i) {
			std::cout << "[" << dimension(i) << "]";
		}
		std::cout << std::endl;
	}


};




#endif /* EXPRESSIONS_BINARY_CORRELATION_H_ */
