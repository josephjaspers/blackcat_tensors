/*
 * Array.h
 *
 *  Created on: Dec 30, 2017
 *      Author: joseph
 */

#ifndef EXPRESSIONS_OPS_H_
#define EXPRESSIONS_OPS_H_

#include "Expression_Binary.h"
#include "Operations/Binary.h"
#include "Operations/Unary.h"

#include "BC_Utility/Determiners.h"
#include "Parse_Tree_Evaluator.h"
//#include "Function_gemm.h"
//#include "Function_transpose.h"
//#include "Parse_Tree_Evaluator.h"

namespace BC {
namespace oper {
template<class> struct dotproduct;
}

namespace internal {

//This is where the beautiful lazy expressions are created

template<class derived
,
template<class,class,class> class expr2 = BC::internal::binary_expression,
template<class,class> class expr1 = BC::internal::unary_expression
>
class expression_interface {

		using mathlib_type 		= _mathlib<derived>;

		//determines the return type of pointwise operations
		template<class param_deriv, class op>
		struct impl {
			using type = 	   expr2<derived ,param_deriv, op>;
			using unary_type = expr1 <derived, op>;
		};

		//determines the return type of dot-product operations (and scalar multiplication)
		template<class param_deriv>
		struct dp_impl {
			static constexpr bool SCALAR_MUL = derived::DIMS() == 0 || param_deriv::DIMS() == 0;

			using dot_type 				= expr2<derived, param_deriv, oper::dotproduct<mathlib_type>>;
			using scalmul_type 			= expr2<derived ,param_deriv, oper::scalar_mul>;

			using type = std::conditional_t<!SCALAR_MUL,
							dot_type,
							scalmul_type>;
		};
	//Returns the class returned as its most derived member
	 const derived& as_derived() const { return static_cast<const derived&>(*this);  }
	 	   derived& as_derived() 	  { return static_cast<	     derived&>(*this); }


	//--------------------------------------evaluation implementation-----------------------------------------------//
	template<bool BARRIER = true, class derived_t>
	void evaluate(const expression_interface<derived_t>& tensor) {
		BC::Evaluator<mathlib_type, BARRIER>::evaluate(as_derived(), tensor.as_derived());
	}

public:

	//--------------------------------------assignment operators-----------------------------------------------//

	template<class pDeriv>
	derived& operator =(const expression_interface<pDeriv>& param) {
		assert_same_size(param);
		evaluate(bi_expr<oper::assign>(param));
		return as_derived();
	}
	template<class pDeriv>
	derived& operator +=(const expression_interface<pDeriv>& param) {
		assert_same_size(param);
		evaluate(bi_expr<oper::add_assign>(param));
		return as_derived();

	}
	template<class pDeriv>
	derived& operator -=(const expression_interface<pDeriv>& param) {
		assert_same_size(param);
		evaluate(bi_expr<oper::sub_assign>(param));
		return as_derived();
	}
	template<class pDeriv>
	derived& operator /=(const expression_interface<pDeriv>& param) {
		assert_same_size(param);
		evaluate(bi_expr<oper::div_assign>(param));
		return as_derived();
	}
	//pointwise multiply
	template<class pDeriv>
	derived& operator %=(const expression_interface<pDeriv>& param) {
		assert_same_size(param);
		evaluate(bi_expr<oper::mul_assign>(param));
		return as_derived();
	}
//	//-------------------------------------dotproduct-------------------- ---------------------//
	template<class pDeriv>
	auto operator *(const expression_interface<pDeriv>& param) const {
		 return typename dp_impl<pDeriv>::type(as_derived(), param.as_derived());
	}
//	//--------------------------------------pointwise operators-------------------------------//
	template<class pDeriv> auto operator +(const expression_interface<pDeriv>& param) const {
		assert_same_size(param);
		return bi_expr<oper::add>(param);
	}
	template<class pDeriv> auto operator -(const expression_interface<pDeriv>& param) const {
		assert_same_size(param);
		return bi_expr<oper::sub>(param);
	}
	template<class pDeriv> auto operator /(const expression_interface<pDeriv>& param) const {
		assert_same_size(param);
		return bi_expr<oper::div>(param);
	}
	//pointwise multiply
	template<class pDeriv> auto operator %(const expression_interface<pDeriv>& param) const {
		assert_same_size(param);
		return bi_expr<oper::mul>(param);
	}


//	 //--------------------------------Other Operators------------------------------//

	 auto operator - () const {
		 return this->un_expr<oper::negation>();
	 }
	template<class pDeriv>
	auto operator ==(const expression_interface<pDeriv>& param) const {
		assert_same_size(param);
		return bi_expr<oper::equal>(param);
	}
	template<class pDeriv>
	auto operator >(const expression_interface<pDeriv>& param) const {
		assert_same_size(param);
		return bi_expr<oper::greater>(param);
	}
	template<class pDeriv>
	auto operator <(const expression_interface<pDeriv>& param) const {
		assert_same_size(param);
		return bi_expr<oper::lesser>(param);
	}
	template<class pDeriv>
	auto operator >=(const expression_interface<pDeriv>& param) const {
		assert_same_size(param);
		return bi_expr<oper::greater_equal>(param);
	}
	template<class pDeriv>
	auto operator <=(const expression_interface<pDeriv>& param) const {
		assert_same_size(param);
		return bi_expr<oper::lesser_equal>(param);
	}
//
//	//-----------------------------------custom expressions--------------------------------------------------//
	template<class functor>
	auto un_expr(functor f) const {
		return typename impl<derived, functor>::unary_type(as_derived(), f);
	}
	template<class functor>
	const auto un_expr() const {
		return typename impl<derived, functor>::unary_type(as_derived());
	}
	template<class d2, class functor>
	const auto bi_expr(functor f, const expression_interface<d2>& rv) const {
		return typename impl<d2, functor>::type(as_derived(), rv.as_derived());
	}
	template<class functor, class d2>
	const auto bi_expr(const expression_interface<d2>& rv) const {
		return typename impl<d2, functor>::type(as_derived(), rv.as_derived());
	}
//	 //--------------------------------ASSERTIONS------------------------------//


	//assert either scalar by tensor operation or same size (not same dimensions)
	template<class deriv>
	void assert_same_size(const expression_interface<deriv>& tensor) const {
		assert_same_ml(tensor);

		if (derived::DIMS() != 0 && deriv::DIMS() != 0)
			if (this->as_derived().DIMS() != 0 && tensor.as_derived().DIMS() != 0)
				if ((as_derived().size() != tensor.as_derived().size())){
					std::cout << "this->DIMS() = " << derived::DIMS() << " this->size() = " <<  as_derived().size() <<  " this_dims ";
					as_derived().print_dimensions();

					std::cout <<  "this->DIMS() = " << deriv::DIMS() << " param.size() = " << tensor.as_derived().size() <<  " param_dims ";
					tensor.as_derived().print_dimensions();
					std::cout << std::endl;

					throw std::invalid_argument("Tensor by Tensor operation - size mismatch - ");
				}
	}

	//assert same math library // asserts both memory is allocated on gpu or cpu
	template<class deriv>
	void assert_same_ml(const expression_interface<deriv>& tensor) const {
		static_assert(std::is_same<_mathlib<derived>, _mathlib<deriv>>::value, "mathlib_type must be identical");
	}
};
}
}

#endif /* TENSOR_CORE_H_ */
