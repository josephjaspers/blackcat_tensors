/*
 * Expressions_Binary_Correlation.h
 *
 *  Created on: Mar 22, 2018
 *      Author: joseph
 */

#ifndef EXPRESSIONS_BINARY_CORRELATION_PADDED_H_
#define EXPRESSIONS_BINARY_CORRELATION_PADDED_H_

#include "Expression_Base.h"
namespace BC {
template<class lv, class rv, int corr_dimension = 2>
struct binary_expression_correlation_padded : expression_base<binary_expression_correlation_padded<lv, rv, corr_dimension>> {

	static_assert(lv::DIMS() == rv::DIMS(), "CORRELATION CURRENTLY ONLY SUPPORTED FOR SAME ORDER TENSORS");
	__BCinline__  static constexpr int DIMS() { return corr_dimension; }
	__BCinline__ static constexpr int CONTINUOUS() { return corr_dimension; }

	using T = _scalar<lv>;

	lv left;  //krnl
	rv right; //img

	binary_expression_correlation_padded(lv l_, rv r_) :left(l_), right(r_) {}

	__BCinline__ const auto innerShape() const {
		return l_array([=](int i) {return right.dimension(i) + left.dimension(i) - 1;} );
	}

	__BCinline__ const auto outerShape() const {
		return l_array([=](int i) {return i == 0 ? this->rows() : this->dimension(i) * this->dimension(i - 1);} );
	}

	template<int mv, class K, class I> __BCinline__
	T axpy(int index, const K& krnl, const I& img) const {

		static_assert(K::DIMS() == I::DIMS(), "Krnl/Img DIMS() must be equal");
		static constexpr int ORDER = K::DIMS() - 1;

		T sum = 0;


		if (mv == 0) {
			if (ORDER == 0)
				for (int i = 0; i < left.rows(); ++i) {
					int img_index = index + i - krnl.rows() + 1;

					if (img_index > -1 and img_index < img.rows())
						sum += krnl[i] * img[img_index];
				}
			else {
				int offset = ((int)(index / this->LD_dimension(ORDER))) - krnl.dimension(ORDER) + 1;
				int index_ = index % this->LD_dimension(ORDER);
				for (int i = 0; i < krnl.dimension(ORDER); ++i) {
					if (i + offset < img.dimension(ORDER))
					sum += axpy<0>(index_, krnl.slice(i), img.slice(i + offset));
				}
			}
		} else {
			int offset = (int)(index / innerShape()[ORDER]) - krnl.dimension(ORDER) + 1;
			int index_ = index % innerShape()[ORDER];

			for (int i = 0; i < krnl.dimension(ORDER); ++i) {
				if (i + offset < img.dimension(ORDER))
				sum += axpy<(((mv - 1) < 0) ? 0 : (mv - 1))>(index_, krnl.slice(i), img.slice(i + offset));
			}
		}

		return sum;
	}

	__BCinline__  auto operator [] (int i) const {
		return axpy<corr_dimension - 1>(i, left, right);
	}


};


}


#endif /* EXPRESSIONS_BINARY_CORRELATION_H_ */
