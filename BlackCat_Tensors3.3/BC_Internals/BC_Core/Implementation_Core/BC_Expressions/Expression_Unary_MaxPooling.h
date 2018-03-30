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

	static constexpr int DIMS() { return ary::DIMS(); }

	stack_array<int, DIMS()> positions;
	stack_array<int, DIMS()> os = init_outerShape();
	ary value;

	unary_expression_maxpooling(ary value_) : value(value_) {
		for (int i = 0; i < DIMS(); ++i) {
			positions[i] = value.dimension(i) - search_space + 1;
		}
	}

	template<int mv, class I>
	T max_pool_search(int index, const I& img) const {
		static constexpr int ORDER = search_space - 1;

		if (mv == 0) {
			std::cout << " mv == 0 " << std::endl;
			if (ORDER == 0) {
				std::cout << " order == 0 " << std::endl;

				T max = img[index];
//				for (int i = 1; i < search_space; ++i)
//					if (img[index + i] > max)
//						max = img[index + i];

				return max;

			} else {
				std::cout << " mv = " << mv <<std::endl;
				std::cout << " order = " << ORDER <<std::endl;

				int offset = (int) (index / LD_dimension(ORDER));
				int index_ = index % LD_dimension(ORDER);
				T max = 0;
				max = max_pool_search<0>(index_, img.slice(offset));
//
//				for (int i = 1; i < search_space; ++i) {
//					auto tmp_max = max_pool_search<0>(index_, img.slice(i + offset));
//					if (tmp_max > max)
//						max = tmp_max;
//				}

				return max;
			}
		} else {

			int offset = (int) (index / positions[ORDER]);
			int index_ = index % positions[ORDER];
			T max = max_pool_search<(((mv - 1) < 0) ? 0 : (mv - 1))>(index_, img.slice(offset));

//			for (int i = 1; i < search_space; ++i) {
//				auto tmp = max_pool_search<(((mv - 1) < 0) ? 0 : (mv - 1))>(index_, img.slice(i + offset));
//
//				if (tmp > max)
//					max = tmp;
//			}
			return max;
		}
	}

	__BCinline__  T operator [] (int i) const {
		std::cout << "max pool --------" << i << std::endl;
		return 0;
//		return max_pool_search<search_space - 1>(i, value);
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
