/*
 * Expressions_Binary_Correlation.h
 *
 *  Created on: Mar 22, 2018
 *      Author: joseph
 */

#ifndef EXPRESSIONS_BINARY_CORRELATION_H_
#define EXPRESSIONS_BINARY_CORRELATION_H_

#include "BlackCat_Internal_Type_ExpressionBase.h"
namespace BC {
template<class lv, class rv, int corr_dimension = 2>
struct binary_expression_correlation : Expression_Core_Base<binary_expression_correlation<lv, rv, corr_dimension>> {

	static_assert(lv::DIMS() == rv::DIMS(), "CORRELATION CURRENTLY ONLY SUPPORTED FOR SAME ORDER TENSORS");
	__BCinline__ static constexpr int DIMS() { return corr_dimension; }
//
	using T = _scalar<lv>;

	lv left;  //krnl
	rv right; //img

	binary_expression_correlation(lv l_, rv r_) :left(l_), right(r_) {}

	__BCinline__ const auto innerShape() const {
		return l_array([=](int i) {return right.dimension(i) - left.dimension(i) + 1;} );
	}

	__BCinline__ const auto outerShape() const {
		return l_array([=](int i) {return i == 0 ? this->rows() : this->dimension(i) * this->dimension(i - 1);} );
	}

	template<int mv, class K, class I>
	T axpy(int index, const K& krnl, const I& img) const {
		index %= img.rows();
		T sum = 0;
		for (int m = 0; m < krnl.rows(); ++m) {
			for (int n = 0; n < krnl.cols(); ++n) {
				sum += krnl[m * krnl.LD_rows() + n] * img[m * img.LD_rows() + n + index];
			}
		}
		return sum;
	}

	__BCinline__  T operator [] (int i) const {
		return axpy<corr_dimension - 1>(i, left, right);
	}

};


}


#endif /* EXPRESSIONS_BINARY_CORRELATION_H_ */
