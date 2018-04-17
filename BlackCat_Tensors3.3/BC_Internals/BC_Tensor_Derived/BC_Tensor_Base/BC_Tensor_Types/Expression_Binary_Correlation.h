/*
 * Expressions_Binary_Correlation.h
 *
 *  Created on: Mar 22, 2018
 *      Author: joseph
 */

#ifndef EXPRESSIONS_BINARY_CORRELATION_H_
#define EXPRESSIONS_BINARY_CORRELATION_H_

#include "BlackCat_Expression_Base.h"
namespace BC {
template<class lv, class rv, int corr_dimension = 2>
struct binary_expression_correlation : Expression_Core_Base<binary_expression_correlation<lv, rv, corr_dimension>> {

	__BCinline__ static constexpr int DIMS() { return corr_dimension; }
	__BCinline__ static constexpr int CONTINUOUS() { return corr_dimension; }


	static_assert(lv::DIMS() == rv::DIMS(), "CORRELATION CURRENTLY ONLY SUPPORTED FOR SAME ORDER TENSORS");
	static_assert(DIMS() <= 3, "CORRELATION MOST MOVEMENT IS LIMITED TO 3D");
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
		T sum = 0;
//
		if (I::DIMS() == K::DIMS()){
			if (mv == 1) {
				for (int i = 0; i < krnl.rows(); ++i)
					sum += krnl[i] * img[i + index];
			}
			if (mv == 2) {
				index %= img.rows();
				for (int n = 0; n < krnl.cols(); ++n)
					for (int m = 0; m < krnl.rows(); ++m)
						sum += krnl[m + n * krnl.LD_rows()] * img[m + n * img.LD_rows() + index];
			}
			if (mv == 3) {
				index %= img.LD_cols(); //is ld_cols is actually size
				index %= img.rows();
				for (int k = 0; k < krnl.dimension(2); ++k)
					for (int n = 0; n < krnl.cols(); ++n)
						for (int m = 0; m < krnl.rows(); ++m)
							sum += krnl[m + n * krnl.LD_rows() + k * krnl.LD_cols()] * img[m + n * img.LD_rows() + k* img.LD_cols() + index];
			}
		}
				return sum;
	}

	__BCinline__  T operator [] (int i) const {
		return axpy<corr_dimension>(i, left, right);
	}

};


}


#endif /* EXPRESSIONS_BINARY_CORRELATION_H_ */
