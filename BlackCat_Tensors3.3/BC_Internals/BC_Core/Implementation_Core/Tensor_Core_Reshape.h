/*
 * Tensor_Reshape.h
 *
 *  Created on: Mar 31, 2018
 *      Author: joseph
 */

#ifndef TENSOR_RESHAPE_H_
#define TENSOR_RESHAPE_H_
#include "BC_Expressions/Expression_Base.h"
#include "Determiners.h"
#include <iostream>
#include "Tensor_Core_Scalar.h"
#include "Tensor_Core_RowVector.h"
namespace BC {

template<class PARENT, int NEW_DIM>
struct Tensor_Reshape : expression<_scalar<PARENT>, Tensor_Reshape<PARENT, NEW_DIM>> {

	using scalar_type = _scalar<PARENT>;
	using self = Tensor_Slice<PARENT>;
	using Mathlib = typename  PARENT::Mathlib;

	__BCinline__ static constexpr int DIMS() { return NEW_DIM; };
	__BCinline__ static constexpr int LAST()  { return NEW_DIM - 1; }
	static_assert(NEW_DIM > -1, "DIMENSIONALITY OF TENSOR MUST BE >= 0");

	using slice_type = Tensor_Slice<self>;

	PARENT parent;
	int* is = Mathlib::unified_initialize(is, DIMS());
	int* os = Mathlib::unified_initialize(os, DIMS());

	operator const PARENT	   () const	{ return parent; }
	operator 	   scalar_type*()       { return parent; }
	operator const scalar_type*() const { return parent; }

	template<int dim> void init() {/*DONT DELETE I HELP COMPILE*/ }

	template<int dim, class... integers>
	void init(int front, integers... ints) {
		is[dim] = front;
		if (dim > 0)
			os[dim] = front * os[dim - 1];
		else
			os[0] = front;

		if (dim != DIMS() - 1) {
			init<(dim + 1 < DIMS() ? dim + 1 : dim)>(ints...);
		}
	}

	template<class... integers>
	Tensor_Reshape(PARENT parent, integers... ints) : parent(parent) {
		static_assert(sizeof...(integers) == DIMS(), "DIMENSIONALITY OF RESHAPE MUST EQUAL THE PARAMETER INTEGERS FOR RESHAPE");
		init<0>(ints...);
		if (this->size() != parent.size()) {
			std::cout << "TENSOR RESHAPE SIZE MUST BE EQUAL TO ITS ORIGINAL SIZE" << std::endl;
			throw std::invalid_argument("INVALID RESHAPE");
		}
	}
	__BCinline__ int dims() const { return DIMS(); }
	__BCinline__ int size() const { return DIMS() > 0 ? outerShape()[LAST()] : 1;  }
	__BCinline__ int rows() const { return DIMS() > 0 ? innerShape()[0] : 1; }
	__BCinline__ int cols() const { return DIMS() > 1 ? innerShape()[1] : 1; }
	__BCinline__ int dimension(int i) const { return DIMS() > i ? innerShape()[i] : 1; }
	__BCinline__ int LD_rows() const { return DIMS() > 0 ? outerShape()[0] : 1; }
	__BCinline__ int LD_cols() const { return DIMS() > 1 ? outerShape()[1] : 1; }
	__BCinline__ int LD_dimension(int i) const { return DIMS() > i + 1 ? outerShape()[i] : 1; }
	__BCinline__ const auto& operator [] (int i) const { return DIMS() == 0 ? parent[0] : parent[i]; }
	__BCinline__ auto& operator [] (int i)  	       { return DIMS() == 0 ? parent[0] : parent[i]; }

	void printDimensions() 		const {
		for (int i = 0; i < DIMS(); ++i)
			std::cout << "[" << dimension(i) << "]";
		std::cout << std::endl;
	}
	void printLDDimensions() const {
		for (int i = 0; i < DIMS(); ++i)
			std::cout << "[" << LD_dimension(i) << "]";
		std::cout << std::endl;

	}

	__BCinline__ const auto innerShape() const 	{ return is; }
	__BCinline__ const auto outerShape() const 	{ return os; }

	__BCinline__
	int slice_index(int i) const {
		if (DIMS() == 0)
			return 0;
		else if (DIMS() == 1)
			return i;
		else
			return outerShape()[LAST() - 1] * i;
	}
	__BCinline__ const auto slice(int i) const { return slice_type(&parent[slice_index(i)],*this); }
	__BCinline__	   auto slice(int i) 	   { return slice_type(&parent[slice_index(i)],*this); }

	__BCinline__ const auto scalar(int i) const { return Tensor_Scalar<self>(&parent[i], *this); }
	__BCinline__ auto scalar(int i) { return Tensor_Scalar<self>(&parent[i], *this); }

	__BCinline__ const auto row(int i) const { return Tensor_Row<self>(&parent[i], *this); }
	__BCinline__ auto row(int i) { return Tensor_Row<self>(&parent[i], *this); }

	__BCinline__ const scalar_type* getIterator() const { return parent.getIterator(); }
	__BCinline__	   scalar_type* getIterator()   	 { return parent.getIterator(); }



	};
}

#endif /* TENSOR_RESHAPE_H_ */
