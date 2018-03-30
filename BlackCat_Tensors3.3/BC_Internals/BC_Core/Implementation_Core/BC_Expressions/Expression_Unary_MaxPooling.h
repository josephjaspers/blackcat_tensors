/*
 * Expressions_Binary_Correlation.h
 *
 *  Created on: Mar 22, 2018
 *      Author: joseph
 */

#ifndef EXPRESSION_UNARY_MAXPOOLING
#define EXPRESSION_UNARY_MAXPOOLING

#include "Expression_Base.h"
namespace BC {
template<class T, class ary,  int search_space = 3>
struct unary_expression_maxpooling : expression<T, unary_expression_maxpooling<T, ary, search_space>> {

	__BCinline__ static constexpr int DIMS() { return ary::DIMS(); }

	stack_array<int, DIMS()> positions;
	stack_array<int, DIMS()> os = init_outerShape();
	ary value;

	unary_expression_maxpooling(ary value_) : value(value_) {
		for (int i = 0; i < DIMS(); ++i) {
			positions[i] = value.dimension(i) - search_space + 1;
		}
	}

	template<int mv, class I> __BCinline__
		T maxp(int index, const I& img) const {

			static constexpr int ORDER = I::DIMS() - 1;

			T sum = 0;

//
//			if (mv == 0) {
//				if (ORDER == 0)
//					for (int i = 0; i < value.rows(); ++i) {
//						sum += img[index + i];
//					}
//				else {
//					int offset = (int)(index / LD_dimension(ORDER));
//					int index_ = index % LD_dimension(ORDER);
//					for (int i = 0; i < value.dimension(ORDER); ++i) {
//						sum += maxp<0>(index_, img.slice(i + offset));
//					}
//				}
//			} else {
//				int offset = (int)(index / positions[ORDER]);
//				int index_ = index % positions[ORDER];
//
//				for (int i = 0; i < value.dimension(ORDER); ++i) {
//						sum += maxp<(((mv - 1) < 0) ? 0 : (mv - 1))>(index, img.slice(i + offset));
//				}
//			}

			return sum;
		}

		__BCinline__  T operator [] (int i) const {
			return 0;//maxp<search_space - 1>(i, value);
		}

	__BCinline__ int size() const {
		int sz = 1;
		for (int i = 0; i < DIMS(); ++i){
			sz *= dimension(i);
		}
		return sz;
	}



	__BCinline__ int dims() const { return DIMS(); }
	__BCinline__ int rows() const { return value.rows() - search_space + 1; };
	__BCinline__ int cols() const { return value.rows() - search_space + 1; };

	__BCinline__ int LD_rows() const { return rows(); }
	__BCinline__ int LD_cols() const { return size(); }
	__BCinline__ int dimension(int i) const { return (value.dimension(i) - search_space + 1); }
	__BCinline__ int LD_dimension(int i) const { return os[i]; }


	__BCinline__ const auto innerShape() const {
		return ref_array(*this);
	}

	__BCinline__ const auto init_outerShape() const {
		stack_array<int, DIMS()> os;
		os[0] = rows();
		for (int i = 1; i < DIMS(); ++i) {
			os[i] = dimension(i) * os[i - 1];
		}
		return os;
	}

	void printDimensions()const  {
		for (int i = 0; i < DIMS(); ++i) {
			std::cout << "[" << dimension(i) << "]";
		}
		std::cout << std::endl;
	}
};
}
#endif /* MAXPOOLING */
