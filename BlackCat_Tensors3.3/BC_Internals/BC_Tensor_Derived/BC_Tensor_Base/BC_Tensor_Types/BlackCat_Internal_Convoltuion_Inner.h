/*
 * BlackCat_Internal_Convoltuion_Inner.h
 *
 *  Created on: Apr 15, 2018
 *      Author: joseph
 */

#ifndef BLACKCAT_INTERNAL_CONVOLTUION_INNER_H_
#define BLACKCAT_INTERNAL_CONVOLTUION_INNER_H_

#include "BlackCat_Internal_Type_ExpressionBase.h" 	//Expression_Core_Base<T, binary_expression_dotproduct<T, lv, rv, Mathlib>> {
#include "BlackCat_Internal_Type_CoreBase.h" 	//Expression_Core_Base<T, binary_expression_dotproduct<T, lv, rv, Mathlib>> {

#include "Expression_Binary_Dotproduct_impl.h"
#include "BlackCat_Internal_Definitions.h"

namespace BC {

/*
 * a = M x K
 * b = K x N
 * c = M x N
 */
//det_Eval

template<class lv, class rv, class Mathlib>
struct binary_expression_convolution_inner : Tensor_Core_Base<binary_expression_convolution_inner<lv, rv, Mathlib>, rv::DIMS()> {

	using scalar_type = _scalar<lv>;

	/*
	 * left = kernel
	 * right = image
	 *
	 * Dimensions must be equal
	 * if left dim is greater (by 1) than right dim
	 * assumes user wishes to convolve a stack of filters
	 */

	__BCinline__ static constexpr int DIMS() { return rv::DIMS(); }
	static_assert (lv::DIMS() == rv::DIMS() || lv::DIMS() == rv::DIMS() + 1, "Dimensions must be same for inner convolution");
	static constexpr bool transA = det_eval<lv>::transposed;
	static constexpr bool transB = det_eval<rv>::transposed;
	static constexpr bool lv_scalar = det_eval<lv>::scalar;
	static constexpr bool rv_scalar = det_eval<rv>::scalar;
	static constexpr bool lv_eval = det_eval<lv>::evaluate;
	static constexpr bool rv_eval = det_eval<rv>::evaluate;

	lv left;
	rv right;

	scalar_type* array_ptr;
	int is[2] { left.rows(), right.cols() };
	int os[2] { left.rows(), left.rows() * right.cols() };

	binary_expression_convolution_inner(lv left, rv right) : left(left), right(right) {
		Mathlib::initialize(array_ptr, this->size());
		eval();
	}

	__BCinline__ const auto& operator [](int index) const  { return array_ptr[index]; }
	__BCinline__ 	   auto& operator [](int index) 		{ return array_ptr[index]; }

	__BCinline__ const auto innerShape() const { return is; }
	__BCinline__ const auto outerShape() const { return os; }

	__BCinline__ int M() const { return left.rows();  }
	__BCinline__ int N() const { return right.cols(); }
	__BCinline__ int K() const { return left.cols();  }

	void destroy() {
		Mathlib::destroy(array_ptr);
	}

	__BCinline__ 	   scalar_type* getIterator() 		{ return array_ptr; }
	__BCinline__ const scalar_type* getIterator() const { return array_ptr; }


public:

	void eval() {


	}
};

}


//		if (transA)
//		std::cout << "A is transposed" << transA << std::endl;
//		if (transB)
//		std::cout <<"B is transposed" << transB << std::endl;
//		if (lv_scalar)
//		std::cout << "A has scalar " <<lv_scalar << std::endl;
//		if (rv_scalar)
//		std::cout <<"B has scalar" << rv_scalar << std::endl;
//		if (lv_eval)
//		std::cout << "A instant eval" <<lv_eval << std::endl;
//		if(rv_eval)
//		std::cout <<"B instant eval " << rv_eval << std::endl;


#endif /* BLACKCAT_INTERNAL_CONVOLTUION_INNER_H_ */
