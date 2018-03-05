
/*
 * BC_Expression_Binary_Pointwise_ScalarL.h
 *
 *  Created on: Dec 2, 2017
 *      Author: joseph
 */
#ifdef  __CUDACC__
#ifndef EXPRESSION_BINARY_POINTWISE_SCALAR_H_
#define EXPRESSION_BINARY_POINTWISE_SCALAR_H_

#include "Expression_Base.cu"
#include "BlackCat_Internal_Definitions.h"

namespace BC {

template<class T, class operation, class lv, class rv>
class binary_expression_scalar_L : expression<T,binary_expression_scalar_L<T, operation, lv, rv>> {
public:

	using this_type = binary_expression_scalar_L<T, operation, lv, rv>;

	operation oper;
	lv left;
	rv right;

	int rank() const { return right.rank(); }
	int rows() const { return right.rows(); };
	int cols() const { return right.cols(); };
	int size() const { return right.size(); };
	int LD_rows() const { return right.LD_rows(); }
	int LD_cols() const { return right.LD_cols(); }
	int dimension(int i)		const { return right.dimension(i); }
	void printDimensions() 		const { right.printDimensions();   }
	void printLDDimensions()	const { right.printLDDimensions(); }
	auto accessor_packet(int index) const { return right.accessor_packet(index); }
	const auto innerShape() const 			{  std::cout << " inner scalar L " << std::endl;  return right.innerShape(); }
	const auto outerShape() const 			{ return right().outerShape(); }

	inline __attribute__((always_inline))  __BC_gcpu__ binary_expression_scalar_L(lv l, rv r) : left(l), right(r) {}
	inline __attribute__((always_inline))  __BC_gcpu__ auto operator [](int index) const { return oper(left[0], right[index]); }
};

template<class T, class operation, class lv, class rv>
class binary_expression_scalar_R : expression<T, binary_expression_scalar_R<T, operation, lv, rv>> {
public:

	using this_type = binary_expression_scalar_R<T, operation, lv, rv>;

	operation oper;
	const lv left;
	const rv right;

	int rank() const { return left.rank(); }
	int rows() const { return left.rows(); };
	int cols() const { return left.cols(); };
	int size() const { return left.size(); };
	int LD_rows() const { return left.LD_rows(); }
	int LD_cols() const { return left.LD_cols(); }
	int dimension(int i)		const { return left.dimension(i); }
	void printDimensions() 		const { left.printDimensions();   }
	void printLDDimensions()	const { left.printLDDimensions(); }
	auto accessor_packet(int index) const { return left.accessor_packet(index); }
	const int* innerShape() const 			{  std::cout << " inner shape - scalr " << std::endl;  return left().innerShape(); }
	const int* outerShape() const 			{ return left().outerShape(); }

	inline __attribute__((always_inline))  __BC_gcpu__ binary_expression_scalar_R(lv l, rv r) : left(l), right(r) {}
	inline __attribute__((always_inline))  __BC_gcpu__ auto operator [](int index) const { return oper(left[index], right[0]);}
};

template<class T, class operation, class lv, class rv>
class binary_expression_scalar_LR : expression<T, binary_expression_scalar_LR<T, operation, lv, rv>> {
public:
		lv left;
		rv right;
		operation oper;

		int rank() const { return left.rank(); }
		int rows() const { return left.rows(); };
		int cols() const { return left.cols(); };
		int size() const { return left.size(); };
		int LD_rows() const { return left.LD_rows(); }
		int LD_cols() const { return left.LD_cols(); }
		int dimension(int i)		const { return left.dimension(i); }
		void printDimensions() 		const { left.printDimensions();   }
		void printLDDimensions()	const { left.printLDDimensions(); }
		auto accessor_packet(int index) const { return left.accessor_packet(index); }
		const int* innerShape() const 			{ return left().innerShape(); }
		const int* outerShape() const 			{ return left().outerShape(); }

		inline __attribute__((always_inline))  __BC_gcpu__ binary_expression_scalar_LR(lv l, rv r) : left(l), right(r) {}
		inline __attribute__((always_inline))  __BC_gcpu__ auto operator [](int index) const { return oper(left[0], right[0]); }
};
}

#endif /* EXPRESSION_BINARY_POINTWISE_SCALAR_H_ */
#endif
