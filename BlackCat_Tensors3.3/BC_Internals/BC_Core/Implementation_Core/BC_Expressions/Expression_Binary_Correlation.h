/*
 * Expressions_Binary_Correlation.h
 *
 *  Created on: Mar 22, 2018
 *      Author: joseph
 */

#ifndef EXPRESSIONS_BINARY_CORRELATION_H_
#define EXPRESSIONS_BINARY_CORRELATION_H_

template<class, int> struct axpy_krnl;

//template<class T>
//struct axpy_krnl<T, 0> {
//
//	template<class K, class I>
//	static T impl(const K& krnl, const I& img, int base_index) {
//		static_assert(K::DIMS() != 0 && I::DIMS() != 0, "meh");
//		T sum = 0;
//		if (K::DIMS() == 1 && I::DIMS() == 1) {
//			for (int i = 0; i < krnl.rows(); ++i)
//				sum += krnl[i] * img[i + base_index];
//			return sum;
//		} else if (K::DIMS() >  ) {
//
//		}
//	}
//};


#include "Expression_Base.h"
namespace BC {
template<class T, class lv, class rv>
struct binary_expression_correlation : expression<T, binary_expression_correlation<T, lv, rv>> {

	static_assert(lv::DIMS() == rv::DIMS(), "CORRELATION CURRENTLY ONLY SUPPORTED FOR SAME ORDER TENSORS");
	static constexpr int DIMS() { return lv::DIMS(); }

	stack_array<int, DIMS()> positions;
	stack_array<int, DIMS()> os = init_outerShape();
	lv left;  //krnl
	rv right; //img

	binary_expression_correlation(lv l_, rv r_) :left(l_), right(r_) {
		for (int i = 0; i < DIMS(); ++i) {
			positions[i] = right.dimension(i) - left.dimension(i) + 1;
		}
	}

	template<class K, class I> __BCinline__
	T axpy(int index, const K& krnl, const I& img) const {

		static_assert(K::DIMS() == I::DIMS(), "Krnl/Img DIMS() must be equal");
		static constexpr int ORDER = K::DIMS() - 1;

		T sum = 0;
		if (ORDER == 0)
			for (int i = 0; i < left.rows(); ++i) {
				sum += krnl[i] * img[index + i];
			}
		else {
			int offset = (int)(index / LD_dimension(ORDER));
			int index_ = index % LD_dimension(ORDER);

			for (int i = 0; i < krnl.dimension(ORDER); ++i) {
					sum += axpy(index_, krnl.slice(i), img.slice(i + offset));
			}
		}

		return sum;
	}

	__BCinline__  T operator [] (int i) const {
		return axpy(i, left, right);
	}

	__BCinline__ int size() const {
		int sz = 1;
		for (int i = 0; i < DIMS() + 1; ++i)
			sz *= dimension(i);
		return sz;
	}



	__BCinline__ int dims() const { return DIMS(); }
	__BCinline__ int rows() const { return right.rows() - left.rows() + 1; };
	__BCinline__ int cols() const { return right.rows() - left.rows() + 1; };

	__BCinline__ int LD_rows() const { return rows(); }
	__BCinline__ int LD_cols() const { return size(); }
	__BCinline__ int dimension(int i) const { return (right.dimension(i) - left.dimension(i) + 1); }
	__BCinline__ int LD_dimension(int i) const { return os[i]; }


	__BCinline__ const auto innerShape() const {
		return ref_array(*this);
	}

	__BCinline__ const auto init_outerShape() const {
		stack_array<int, DIMS()> ary;
		ary[0] = rows();
		for (int i = 1; i < DIMS(); ++i) {
			ary[i] = dimension(i) * ary[i - 1];
		}
		return ary;
	}

	void printDimensions()const  {
		for (int i = 0; i < DIMS(); ++i) {
			std::cout << "[" << dimension(i) << "]";
		}
		std::cout << std::endl;
	}
};


}


#endif /* EXPRESSIONS_BINARY_CORRELATION_H_ */
