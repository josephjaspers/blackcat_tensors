/*
 * Array.h
 *
 *  Created on: Dec 30, 2017
 *      Author: joseph
 */

#ifndef TENSOR_HEAD_H_
#define TENSOR_HEAD_H_

#include "Expression_Templates/Expression_Binary.h"
#include "Expression_Templates/Operations/Binary.h"
#include "Expression_Templates/Operations/Unary.h"

#include "Expression_Templates/Function_conv2.h"

#include "Expression_Templates/Function_gemm.h"
#include "Expression_Templates/Function_gemv.h"
#include "Expression_Templates/Function_ger.h"
#include "Expression_Templates/Function_dot.h"

#include "Expression_Templates/Expression_Unary.h"
#include "Expression_Templates/Function_transpose.h"

//#include "Tensor_Operations_Impl/Expression_Determiner.h"
#include "Tensor_Operations_Impl/Alias.h"

#include "Expression_Templates/Parse_Tree_Evaluator.h"

namespace BC {
namespace Base {

//This is where the beautiful lazy expressions are created

template<class derived>
class Tensor_Operations {

	template<class> friend class Tensor_Operations;

	using functor_type 		= functor_of<derived>;
	using scalar_type 		= scalar_of<derived>;
	using mathlib_type 		= mathlib_of<derived>;

	template<class expr> 		   using unary_expression_t  = BC::Tensor_Base<internal::unary_expression<functor_type, expr>>;
	template<class rv, class expr> using binary_expression_t = BC::Tensor_Base<internal::binary_expression<functor_type, rv, expr>>;

	const derived& as_derived() const { return static_cast<const derived&>(*this); }
	 	  derived& as_derived() 	  { return static_cast<	     derived&>(*this); }

	//--------------------------------------evaluation implementation-----------------------------------------------//
	template<class derived_t>
	void evaluate(const Tensor_Operations<derived_t>& tensor) {
		BC::Evaluator<mathlib_type>::evaluate(tensor.as_derived().internal());
	}

public:

	//--------------------------------------assignment operators-----------------------------------------------//
	template<class pDeriv>
	derived& operator =(const Tensor_Operations<pDeriv>& param) {
		BC_ARRAY_ONLY("derived& operator =(const Tensor_Operations<pDeriv>& param)");
		assert_valid(param);
		evaluate(bi_expr<oper::assign>(param));
		return as_derived();
	}
	template<class pDeriv>
	derived& operator +=(const Tensor_Operations<pDeriv>& param) {
		BC_ARRAY_ONLY("derived& operator +=(const Tensor_Operations<pDeriv>& param)");
		assert_valid(param);
		evaluate(bi_expr<oper::add_assign>(param));
		return as_derived();
	}
	template<class pDeriv>
	derived& operator -=(const Tensor_Operations<pDeriv>& param) {
		BC_ARRAY_ONLY("derived& operator -=(const Tensor_Operations<pDeriv>& param)");
		assert_valid(param);
		evaluate(bi_expr<oper::sub_assign>(param));
		return as_derived();
	}
	template<class pDeriv>
	derived& operator /=(const Tensor_Operations<pDeriv>& param) {
		BC_ARRAY_ONLY("derived& operator /=(const Tensor_Operations<pDeriv>& param)");
		assert_valid(param);
		evaluate(bi_expr<oper::div_assign>(param));
		return as_derived();
	}
	//pointwise multiply
	template<class pDeriv>
	derived& operator %=(const Tensor_Operations<pDeriv>& param) {
		BC_ARRAY_ONLY("derived& operator %=(const Tensor_Operations<pDeriv>& param)");
		assert_valid(param);
		evaluate(bi_expr<oper::mul_assign>(param));
		return as_derived();
	}
	//-------------------------------------gemm/gemv/ger-----------------------------------------//
	template<class param_deriv>
	auto operator *(const Tensor_Operations<param_deriv>& param) const {

		static constexpr bool scalmul	= derived::DIMS() == 0 || param_deriv::DIMS() == 0;
		static constexpr bool gemm 		= derived::DIMS() == 2 && param_deriv::DIMS() == 2;
		static constexpr bool gemv 		= derived::DIMS() == 2 && param_deriv::DIMS() == 1;
		static constexpr bool ger  		= derived::DIMS() == 1 && param_deriv::DIMS() == 1 && internal::det_eval<param_deriv>::transposed;
		static constexpr bool dot		= derived::DIMS() == 1 && param_deriv::DIMS() == 1 && !ger;
		using matmul_t =
					 std::conditional_t<scalmul, binary_expression_t<functor_of<param_deriv>, oper::scalar_mul>,
					 std::conditional_t<gemm, 	 binary_expression_t<functor_of<param_deriv>, oper::gemm<mathlib_type>>,
					 std::conditional_t<gemv, 	 binary_expression_t<functor_of<param_deriv>, oper::gemv<mathlib_type>>,
					 std::conditional_t<ger, 	 binary_expression_t<functor_of<param_deriv>, oper::ger<mathlib_type>>,
					 std::conditional_t<dot,	 binary_expression_t<functor_of<param_deriv>, oper::dot<mathlib_type>>, void>>>>>;

		static_assert(!std::is_same<matmul_t, void>::value, "Matrix Multiplication currently does not support broadcasting");

		return matmul_t(as_derived().internal(), param.as_derived().internal());
	}

	//--------------------------------------pointwise operators-------------------------------//
	template<class pDeriv> auto operator +(const Tensor_Operations<pDeriv>& param) const {
		assert_valid(param);
		return bi_expr<oper::add>(param);
	}
	template<class pDeriv> auto operator -(const Tensor_Operations<pDeriv>& param) const {
		assert_valid(param);
		return bi_expr<oper::sub>(param);
	}
	template<class pDeriv> auto operator /(const Tensor_Operations<pDeriv>& param) const {
		assert_valid(param);
		return bi_expr<oper::div>(param);
	}
	//pointwise multiply
	template<class pDeriv> auto operator %(const Tensor_Operations<pDeriv>& param) const {
		assert_valid(param);
		return bi_expr<oper::mul>(param);
	}


	 //--------------------------------Other Operators------------------------------//

	 auto operator - () const {
		 return un_expr<oper::negation>();
	 }
	template<class pDeriv>
	auto operator ==(const Tensor_Operations<pDeriv>& param) const {
		assert_valid(param);
		return bi_expr<oper::equal>(param);
	}
	template<class pDeriv>
	auto operator >(const Tensor_Operations<pDeriv>& param) const {
		assert_valid(param);
		return bi_expr<oper::greater>(param);
	}
	template<class pDeriv>
	auto operator <(const Tensor_Operations<pDeriv>& param) const {
		assert_valid(param);
		return bi_expr<oper::lesser>(param);
	}
	template<class pDeriv>
	auto operator >=(const Tensor_Operations<pDeriv>& param) const {
		assert_valid(param);
		return bi_expr<oper::greater_equal>(param);
	}
	template<class pDeriv>
	auto operator <=(const Tensor_Operations<pDeriv>& param) const {
		assert_valid(param);
		return bi_expr<oper::lesser_equal>(param);
	}

	//alias ----------------------
	template<class alias> friend class Alias;
	Alias<derived> alias() {
		return Alias<derived>(as_derived());
	}


	template<int x, class param_derived> auto conv(const Tensor_Operations<param_derived>& tensor) const {
		return as_derived().bi_expr<oper::conv<x, mathlib_type>>(tensor.as_derived());
	}
	//-----------------------------------custom expressions--------------------------------------------------//

	template<class functor>
	auto un_expr(functor f) const {
		return unary_expression_t<functor>(as_derived().internal(), f);
	}
	template<class functor>
	const auto un_expr() const {
		return unary_expression_t<functor>(as_derived().internal());
	}
	template<class functor, class right_value>
	const auto bi_expr(functor f, const Tensor_Operations<right_value>& rv) const {
		return binary_expression_t<functor_of<right_value>, functor>(as_derived().internal(), rv.as_derived().internal());
	}
	template<class functor, class right_value>
	const auto bi_expr(const Tensor_Operations<right_value>& rv) const {
		return binary_expression_t<functor_of<right_value>, functor>(as_derived().internal(), rv.as_derived().internal());
	}
	 //--------------------------------ASSERTIONS------------------------------//


	template<class deriv> __BC_host_inline__ bool non_scalar_op(const Tensor_Operations<deriv>& tensor) const {
		return derived::DIMS() != 0 && deriv::DIMS() != 0;
	}
	template<class deriv> __BC_host_inline__ bool same_rank(const Tensor_Operations<deriv>& tensor) const {
		return derived::DIMS() == deriv::DIMS();
	}
	template<class deriv> __BC_host_inline__ bool same_size(const Tensor_Operations<deriv>& tensor) const {
		return this->as_derived().size() == tensor.as_derived().size();
	}

	//ensures that the smaller tensor is a same-dimensioned "slice" of the other
	template<class deriv> __BC_host_inline__ bool valid_slice(const Tensor_Operations<deriv>& tensor) const {
		constexpr int DIM_MIN = MTF::min(derived::DIMS(), deriv::DIMS());
		for (int i = 0; i < DIM_MIN; ++i)
			if (tensor.as_derived().dimension(i) != as_derived().dimension(i))
				return false;

		return true;
	}

	template<class deriv> __BC_host_inline__ bool error_message(const Tensor_Operations<deriv>& tensor) const {
		std::cout << "this->DIMS() = " << derived::DIMS() << " this->size() = " <<  as_derived().size() <<  " this_dims ";
		as_derived().print_dimensions();
		std::cout <<  "param->DIMS() = " << deriv::DIMS() << " param.size() = " << tensor.as_derived().size() <<  " param_dims ";
		tensor.as_derived().print_dimensions();
		std::cout << std::endl;
		throw std::invalid_argument("Tensor by Tensor operation - size mismatch - ");
	}

	template<class deriv> __BC_host_inline__
	void assert_valid(const Tensor_Operations<deriv>& tensor) const {
//#ifdef NDEBUG
		assert_same_ml(tensor);						//static_assert same allocation (gpu/cpu)
		if (non_scalar_op(tensor)) {				//check if a tensor by scalar operation
			if (same_rank(tensor)) {				//else check is same dimension (element-wise function) (
				if (!same_size(tensor))					//if is same dimension, ensure same size
					error_message(tensor);				//else error
				} else if (!valid_slice(tensor)) {	//if not same dimension check if valid slice operation
					error_message(tensor);			//else error
				}
		}

//#endif
	}

	//assert same math library // asserts both memory is allocated on gpu or cpu
	template<class deriv>__BC_host_inline__
	void assert_same_ml(const Tensor_Operations<deriv>& tensor) const {
		static_assert(std::is_same<mathlib_of<derived>, mathlib_of<deriv>>::value, "mathlib_type must be identical");
	}
};

}
}

#include "Tensor_Functions/Unary_Functions.h"

#endif /* TENSOR_CORE_H_ */
