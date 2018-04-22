/*
 * Expressions_Binary_Correlation.h
 *
 *  Created on: Mar 22, 2018
 *      Author: joseph
 */

#ifndef EXPRESSIONS_BINARY_CORRELATION_PADDED_H_
#define EXPRESSIONS_BINARY_CORRELATION_PADDED__H_

#include "Expression_Base.h"
#include "Expression_Binary_Pointwise.h"

namespace BC {

template<int dimension = 2> struct _x_corr_padded;

template<class lv, class rv, int corr_dimension>
struct binary_expression<lv, rv, _x_corr_padded<corr_dimension>> : expression_base<binary_expression<lv, rv, _x_corr_padded<corr_dimension>>> {

	__BCinline__ static constexpr int DIMS() { return corr_dimension; }
	__BCinline__ static constexpr int CONTINUOUS() { return corr_dimension; }
	using scalar = _scalar<lv>;

	static_assert(lv::DIMS() == rv::DIMS(), "CORRELATION CURRENTLY ONLY SUPPORTED FOR SAME ORDER TENSORS");
	static_assert(DIMS() <= 3, "CORRELATION MOST MOVEMENT IS LIMITED TO 3D");


	lv left;  //krnl
	rv right; //img

	binary_expression(lv l_, rv r_) :left(l_), right(r_) {}

	__BCinline__ const auto innerShape() const { return l_array([=](int i) {return right.dimension(i) + left.dimension(i) - 1;} ); }
	__BCinline__ const auto outerShape() const { return l_array([=](int i) {return i == 0 ? this->rows() : this->dimension(i) * this->dimension(i - 1);} );}

	struct DISABLED;
	template<int x>
	using conditional_int = std::conditional_t<x == lv::DIMS(), int, DISABLED>;

	//1d correlation
	template<class... ints> __BCinline__
	scalar axpy (conditional_int<1> x, ints... location) const {

		scalar sum = 0;
		for (int i = 0 ; i < left.rows(); ++i) {

			int row_index = i + x - left.rows() + 1;
			if (row_index >= 0 && row_index < right.rows())
				sum += left(i) * right(row_index, location...);
		}
		return sum;
	}
	//2d correlation
	template<class... ints> __BCinline__
	scalar axpy (conditional_int<2> x, int y, ints... indexes) const {

		scalar sum = 0;
		for (int n = 0; n < left.cols(); ++n) {

			int col_index = n + y - left.cols() + 1;
			if (col_index >= 0 && col_index < right.cols())
				for (int m = 0 ; m < left.rows(); ++m) {

					int row_index = m + x - left.rows() + 1;
					if(row_index >= 0 && row_index < right.rows())
						sum += left(m, n) * right(row_index, col_index, indexes...);
				}
		}
		return sum;
	}
	//3d correlation
	template<class... ints> __BCinline__
	scalar axpy (conditional_int<3> x, int y, int z, ints... indexes) const {
		scalar sum = 0;
		for (int k = 0; k < left.dimension(2); ++k) {

			int page_index = k + z - left.dimension(2) + 1;
			if (page_index >= 0)
				for (int n = 0; n < left.cols(); ++n) {

					int col_index = n + y - left.cols() + 1;
					if (col_index >= 0 && col_index < right.cols())
						for (int m = 0 ; m < left.rows(); ++m) {

							int row_index = m + x - left.rows() + 1;
							if(row_index >= 0 && row_index < right.rows())
								sum += left(m, n, k) * right(row_index, col_index, page_index, indexes...);
						}
				}
		}
		return sum;
	}


	template<class... integers> __BCinline__  scalar operator ()(integers... ints) const {
		return axpy(ints...);
	}
};
}


#endif /* EXPRESSIONS_BINARY_CORRELATION_H_ */

