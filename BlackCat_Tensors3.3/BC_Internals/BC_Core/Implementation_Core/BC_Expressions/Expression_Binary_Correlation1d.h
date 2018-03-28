/*
 * Expression_Binary_Correlation1d.h
 *
 *  Created on: Mar 27, 2018
 *      Author: joseph
 */

#ifndef EXPRESSION_BINARY_CORRELATION1D_H_
#define EXPRESSION_BINARY_CORRELATION1D_H_

#include "Expression_Base.h"
namespace BC {
template<class T, class lv, class rv>
struct binary_expression_correlation_1d : expression<T, binary_expression_correlation_1d<T, lv, rv>> {

	static_assert(lv::DIMS() == rv::DIMS(), "CORRELATION CURRENTLY ONLY SUPPORTED FOR SAME ORDER TENSORS");
	static constexpr int DIMS() { return 1; }

	lv left;  //krnl
	rv right; //img

	binary_expression_correlation_1d(lv l_, rv r_) :left(l_), right(r_) {}

	template<class K, class I> __BCinline__
	T axpy(int index, const K& krnl, const I& img) const {

		static_assert(K::DIMS() == I::DIMS(), "Krnl/Img DIMS() must be equal");
		static constexpr int ORDER = K::DIMS() - 1;

		T sum = 0;
			if (ORDER == 0) {
			for (int i = 0; i < left.rows(); ++i) {
				sum += krnl[i] * img[index + i];
			}
		} else {
			int offset = (int) (index / positions[ORDER]);
			int index_ = index % positions[ORDER];

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
		for (int i = 0; i < movements; ++i)
			sz *= dimension(i);
		return sz;
	}
	__BCinline__ int dims() const { return DIMS(); }
	__BCinline__ int rows() const { return right.rows() - left.rows() + 1; };
	__BCinline__ int cols() const { return right.cols() - left.cols() + 1; };

	__BCinline__ int LD_rows() const { return rows(); }
	__BCinline__ int LD_cols() const { return size(); }
	__BCinline__ int dimension(int i) const { return (right.dimension(i) - left.dimension(i) + 1); }
	__BCinline__ int LD_dimension(int i) const { return i == 0 ? rows() : 0; }

	__BCinline__ const auto innerShape() const {
		return ref_array(*this);
	}

	__BCinline__ const auto outerShape() const {
		stack_array<int, DIMS()> ary;
		ary[0] = rows();
		for (int i = 1; i < DIMS(); ++i) {
			ary[i] = dimension(i) * ary[i - 1];
		}
		return ary;
	}

	template<class v, class alt>
	using expr_type = std::conditional_t<v::RANK() == 0, v, alt>;

	__BCinline__ const auto slice(int i) const {
		std::cout << " correlation of slice is not well defined " << std::endl;
		return binary_expression_correlation_1d<T, lv, decltype(right.slice(0))>(left.slice(i), right.slice(i));
	}
	__BCinline__ const auto row(int i) const {
		std::cout << " correlation of slice is not well defined " << std::endl;
		return binary_expression_correlation_1d<T, lv, expr_type<rv, decltype(right.row(0))>>(left.row(i), right.row(i)); }

	__BCinline__ const auto col(int i) const {
		std::cout << " correlation of slice is not well defined " << std::endl;
		return binary_expression_correlation_1d<T, lv, expr_type<rv, decltype(right.col(0))>>(left.col(i), right.col(i)); }


	void printDimensions()const  {
		for (int i = 0; i < DIMS(); ++i) {
			std::cout << "[" << dimension(i) << "]";
		}
		std::cout << std::endl;
	}
	void printLDDimensions()const  {
		auto os = outerShape();
		for (int i = 0; i < DIMS(); ++i) {
			std::cout << "[" << os[i] << "]";
		}
		std::cout << std::endl;
	}
};
//
//template<class T, class lv, class rv>
//struct binary_expression_correlation_1d_padded : expression<T, binary_expression_correlation_1d_padded<T, lv, rv>> {
//
//	static_assert(lv::DIMS() == rv::DIMS(), "CORRELATION CURRENTLY ONLY SUPPORTED FOR SAME ORDER TENSORS");
//	static constexpr int DIMS() { return lv::DIMS(); }
//
//	stack_array<int, DIMS()> positions;
//	lv left;  //krnl
//	rv right; //img
//
//	binary_expression_correlation_1d_padded(lv l_, rv r_) :left(l_), right(r_) {
//		for (int i = 0; i < DIMS(); ++i) {
//			positions[i] = right.dimension(i) + left.dimension(i) - 1;
//		}
//	}
//	template<int curr_dim>
//	int invert(int i) const {
//		return i - (left.dimension(curr_dim) - 1);
//	}
//
//	template<class K, class I> __BCinline__
//	T axpy(int index, const K& krnl, const I& img) const {
//
//		static_assert(K::DIMS() == I::DIMS(), "Krnl/Img DIMS() must be equal");
//		static constexpr int ORDER = K::DIMS() - 1;
//
//		T sum = 0;
//		if (ORDER == 0)
//			for (int i = 0; i <left.rows(); ++i) {
//				if (i + index - (krnl.rows() - 1) > -1 && i + index - (krnl.rows() - 1) < img.rows())
//				sum += krnl[i] * img[i + index - (krnl.rows() - 1)];
//			}
//		else {
//			int img_slice_offset = (int)(index / positions[ORDER]);
//			int index_ = index % positions[ORDER];
//
//			for (int i = 0; i < krnl.dimension(ORDER); ++i) {
//				if (invert<ORDER>(i) + img_slice_offset > -1 && invert<ORDER>(i) + img_slice_offset > -1 < img.dimension(ORDER))
//					sum += axpy(index_, krnl.slice(i), img.slice(invert<ORDER>(i) + img_slice_offset));
//			}
//		}
//
//		return sum;
//	}
//
//	__BCinline__  T operator [] (int i) const {
//		return axpy(i, left, right);
//	}
//
//	int size() const {
//		int sz = 1;
//		for (int i = 0; i < DIMS() + 1; ++i)
//			sz *= dimension(i);
//		return sz;
//	}
//
//
//
//	__BCinline__ int dims() const { return DIMS(); }
//	__BCinline__ int rows() const { return right.rows() + left.rows() - 1; };
//	__BCinline__ int cols() const { return right.rows() + left.rows() - 1; };
//
//	__BCinline__ int LD_rows() const { return rows(); }
//	__BCinline__ int LD_cols() const { return size(); }
//	__BCinline__ int dimension(int i) const { return (right.dimension(i) + left.dimension(i) - 1); }
//	__BCinline__ int LD_dimension(int i) const { return outerShape()[i]; }
//
//
//	__BCinline__ const auto innerShape() const {
//		return ref_array(*this);
//	}
//
//	__BCinline__ const auto outerShape() const {
//		stack_array<int, DIMS()> ary;
//		ary[0] = rows();
//		for (int i = 1; i < DIMS(); ++i) {
//			ary[i] = dimension(i) * ary[i - 1];
//		}
//		return ary;
//	}
//
//	template<class v, class alt>
//	using expr_type = std::conditional_t<v::RANK() == 0, v, alt>;
//
//	__BCinline__ const auto slice(int i) const {
//		std::cout << " correlation of slice is not well defined " << std::endl;
//		return binary_expression_correlation<T, lv, decltype(right.slice(0)>(left.slice(i), right.slice(i));
//	}
//	__BCinline__ const auto row(int i) const {
//		std::cout << " correlation of slice is not well defined " << std::endl;
//		return binary_expression_correlation<T, lv, expr_type<rv, decltype(right.row(0))>(left.row(i), right.row(i)); }
//
//	__BCinline__ const auto col(int i) const {
//		std::cout << " correlation of slice is not well defined " << std::endl;
//		return binary_expression_correlation<T, lv, expr_type<rv, decltype(right.col(0))>>(left.col(i), right.col(i)); }
//
//
//	void printDimensions()const  {
//		for (int i = 0; i < DIMS(); ++i) {
//			std::cout << "[" << dimension(i) << "]";
//		}
//		std::cout << std::endl;
//	}
//};

}



#endif /* EXPRESSION_BINARY_CORRELATION1D_H_ */
