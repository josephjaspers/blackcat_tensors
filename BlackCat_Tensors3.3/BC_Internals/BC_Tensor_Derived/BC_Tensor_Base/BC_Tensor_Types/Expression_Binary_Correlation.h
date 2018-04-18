/*
 * Expressions_Binary_Correlation.h
 *
 *  Created on: Mar 22, 2018
 *      Author: joseph
 */

#ifndef EXPRESSIONS_BINARY_CORRELATION_H_
#define EXPRESSIONS_BINARY_CORRELATION_H_

#include "Expression_Base.h"
#include <functional>
namespace BC {
template<class lv, class rv, int corr_dimension = 2>
struct binary_expression_correlation : expression_base<binary_expression_correlation<lv, rv, corr_dimension>> {

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

	template<int krnl_dims>
	T krnl_corr(int rv_index) {
		T sum = 0;
		int lv_index = 0;

		auto Func = [&](int D) {
			if (D == 1) {
				for (int m = 0; m < left.rows(); ++m) {
					sum += left[lv_index + m] * right[m + rv_index];
				}
			} else {
				for (int m = 0; m < left.dimension(D - 1); ++m) {
					lv_index += left.dimension(D - 1);
				}
			}
		};

	}

	T corr(int x, int y, int z) const {
		static constexpr int imglast = rv::DIMS() - 1;

		for (int k = 0; k < right.dimension(imglast); ++k) {
			for (int n = 0; n < right.dimension(imglast - 1); ++n){
				int rv_index = right.dimension(imglast-1) * right.LD_dimension(imglast-1);
				for (int m = 0; m < right.dimension(imglast - 2); ++m){
					int rv_i;
					if (rv::DIMS() - 2 <= 0)
						rv_i = rv_index + m;
					else
						rv_i = rv_index + m*right.LD_dimension(imglast - 2);
				}
			}
		}
	}


	template<class... integers> __BCinline__
	  T operator () (integers... ints) const {
		static_assert(sizeof...(integers) == corr_dimension, "INTEGER/TENSOR-DIM MISMATCH");
//
	}

};


}


#endif /* EXPRESSIONS_BINARY_CORRELATION_H_ */
