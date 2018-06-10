/*
 * Core.h
 *
 *  Created on: Dec 30, 2017
 *      Author: joseph
 */

#ifndef TENSOR_HEAD_H_
#define TENSOR_HEAD_H_

#include "BC_Tensor_Types/Expression_Binary.h"
#include "BC_Tensor_Types/Operations/Binary.h"
#include "BC_Tensor_Types/Operations/Unary.h"

#include "BC_Tensor_Types/gemm.h"
#include "BC_Tensor_Types/Expression_Unary.h"
#include "BC_Tensor_Types/transpose.h"
#include "Tensor_Operations_Impl/AlternateAsterixDenoter.h"
#include "BC_Tensor_Types/Expression_Determiner.h"
#include "BC_Tensor_Types/BLAS_Injection_Runner.h"

#include "Tensor_Functions/Unary_Functions.h"
#include <type_traits>

namespace BC {
namespace Base {

//This is where the beautiful lazy expressions are created

template<class derived>
class Tensor_Operations {

	template<class> friend class Tensor_Operations;
	template<class pderiv, class functor> using impl 	= typename operationImpl::expression_determiner<derived>::template impl<pderiv, functor>;
	template<class pderiv> 				  using dp_impl	= typename operationImpl::expression_determiner<derived>::template dp_impl<pderiv>;

	using functor_type 		= _functor<derived>;
	using scalar_type 		= _scalar<derived>;
	using mathlib_type 		= _mathlib<derived>;

	//Returns the class returned as its most derived member
	 const derived& as_derived() const { return static_cast<const derived&>(*this);  }
	 	   derived& as_derived() 	  { return static_cast<	     derived&>(*this); }
public:

	//-------------------------------------dotproduct-------------------- ---------------------//
	template<class pDeriv>
	auto operator *(const Tensor_Operations<pDeriv>& param) const {
		 return typename dp_impl<pDeriv>::type(as_derived().internal(), param.as_derived().internal());
	}
	//--------------------------------------pointwise operators-------------------------------//
	template<class pDeriv> auto operator +(const Tensor_Operations<pDeriv>& param) const {
		assert_same_size(param);
		return bi_expr<oper::add>(param);
	}
	template<class pDeriv> auto operator -(const Tensor_Operations<pDeriv>& param) const {
		assert_same_size(param);
		return bi_expr<oper::sub>(param);
	}
	template<class pDeriv> auto operator /(const Tensor_Operations<pDeriv>& param) const {
		assert_same_size(param);
		return bi_expr<oper::div>(param);
	}
	template<class pDeriv> auto operator %(const Tensor_Operations<pDeriv>& param) const {			//overloaded for pointwise multiply
		assert_same_size(param);
		return bi_expr<oper::mul>(param);
	}
	template<class pDeriv> auto operator *(const alternate_asterix_denoter<pDeriv>& param) const { //alternative for pointwise multiply
		assert_same_size(param.get());
		return bi_expr<oper::mul>(param.get());
	}

private:

	//--------------------------------------assignment implementation-----------------------------------------------//
	template<bool BARRIER = true, class derived_t>
	void evaluate(const Tensor_Operations<derived_t>& tensor) {
		BC::Evaluator<mathlib_type, BARRIER>::evaluate(as_derived().internal(), tensor.as_derived().internal());
	}

public:
	//--------------------------------------assignment operators-----------------------------------------------//

	template<class pDeriv>
	derived& operator =(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		evaluate(bi_expr<oper::assign>(param));
		return as_derived();
	}


	template<class pDeriv>
	derived& operator +=(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		evaluate(bi_expr<oper::add_assign>(param));
		return as_derived();

	}
	template<class pDeriv>
	derived& operator -=(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		evaluate(bi_expr<oper::sub_assign>(param));
		return as_derived();
	}
	template<class pDeriv>
	derived& operator /=(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		evaluate(bi_expr<oper::div_assign>(param));
		return as_derived();
	}
	template<class pDeriv>
	derived& operator %=(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		evaluate(bi_expr<oper::mul_assign>(param));
		return as_derived();
	}

	//--------------------------------------assignment operators NO BARRIER-----------------------------------------------//

	template<class pDeriv>
	derived& operator =(const unsafe_AAD<pDeriv>& param) {
		assert_same_size(param.get());
		evaluate<false>(bi_expr<oper::assign>(param.get()));
		return as_derived();
	}


	template<class pDeriv>
	derived& operator +=(const unsafe_AAD<pDeriv>& param) {
		assert_same_size(param.get());
		evaluate<false>(bi_expr<oper::add_assign>(param.get()));
		return as_derived();

	}
	template<class pDeriv>
	derived& operator -=(const unsafe_AAD<pDeriv>& param) {
		assert_same_size(param.get());
		evaluate<false>(bi_expr<oper::sub_assign>(param.get()));
		return as_derived();
	}
	template<class pDeriv>
	derived& operator /=(const unsafe_AAD<pDeriv>& param) {
		assert_same_size(param.get());
		evaluate<false>(bi_expr<oper::div_assign>(param.get()));
		return as_derived();
	}
	template<class pDeriv>
	derived& operator %=(const unsafe_AAD<pDeriv>& param) {
		assert_same_size(param.get());
		evaluate<false>(bi_expr<oper::mul_assign>(param.get()));
		return as_derived();
	}
	 //--------------------------------Other Operators------------------------------//

	 auto operator - () const {
		 return this->un_expr<oper::negation>();
	 }
	template<class pDeriv>
	auto operator ==(const Tensor_Operations<pDeriv>& param) const {
		assert_same_size(param);
		return bi_expr<oper::equal>(param);
	}
	template<class pDeriv>
	auto operator >(const Tensor_Operations<pDeriv>& param) const {
		assert_same_size(param);
		return bi_expr<oper::greater>(param);
	}
	template<class pDeriv>
	auto operator <(const Tensor_Operations<pDeriv>& param) const {
		assert_same_size(param);
		return bi_expr<oper::lesser>(param);
	}
	template<class pDeriv>
	auto operator >(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return bi_expr<oper::max>(param.get());
	}
	template<class pDeriv>
	auto operator <(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return bi_expr<oper::min>(param.get());
	}
	template<class pDeriv>
	auto operator >=(const Tensor_Operations<pDeriv>& param) const {
		assert_same_size(param);
		return bi_expr<oper::greater_equal>(param);
	}
	template<class pDeriv>
	auto operator <=(const Tensor_Operations<pDeriv>& param) const {
		assert_same_size(param);
		return bi_expr<oper::lesser_equal>(param);
	}

	//-------------------------------------------DELAYED ASSIGNMENT OPERATORS---------------------------------------------//

	template<class pDeriv>
	auto operator =(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return bi_expr<oper::assign>(param.get());
	}

	template<class pDeriv>
	auto operator +=(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return bi_expr<oper::add_assign>(param.get());
	}
	template<class pDeriv>
	auto operator -=(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return bi_expr<oper::sub_assign>(param.get());
	}
	template<class pDeriv>
	auto operator %=(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return bi_expr<oper::mul_assign>(param.get());
	}
	template<class pDeriv>
	auto operator *=(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return bi_expr<oper::mul_assign>(param.get());
	}
	template<class pDeriv>
	auto operator /=(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return bi_expr<oper::div_assign>(param.get());
	}

	//-----------------------------------COMBINE EXPRESSION-------------------------------------------------//
	template<class pDeriv> auto operator &&(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		return bi_expr<oper::combine>(param);
	}


	//-----------------------------------custom expressions--------------------------------------------------//
	template<class functor>
	auto un_expr(functor f) const {
		return typename impl<derived, functor>::unary_type(as_derived().internal(), f);
	}
	template<class functor>
	const auto un_expr() const {
		return typename impl<derived, functor>::unary_type(as_derived().internal());
	}
	template<class d2, class functor>
	const auto bi_expr(functor f, const Tensor_Operations<d2>& rv) const {
		return typename impl<d2, functor>::type(as_derived().internal(), rv.as_derived().internal());
	}
	template<class functor, class d2>
	const auto bi_expr(const Tensor_Operations<d2>& rv) const {
		return typename impl<d2, functor>::type(as_derived().internal(), rv.as_derived().internal());
	}
	//------------------------------------alternate asterix denoter----------------------------------//
	 const alternate_asterix_denoter<derived> operator * () const {
		return alternate_asterix_denoter<derived>(*this);
	}

	 auto alias() const {
		 return tensor_alias(*this);
	 }

	 //--------------------------------ASSERTIONS------------------------------//


	//assert either scalar by tensor operation or same size (not same dimensions)
	template<class deriv>
	void assert_same_size(const Tensor_Operations<deriv>& tensor) const {
		assert_same_ml(tensor);

		if (derived::DIMS() != 0 && deriv::DIMS() != 0)
			if (this->as_derived().DIMS() != 0 && tensor.as_derived().DIMS() != 0)
				if ((as_derived().size() != tensor.as_derived().size())){
					std::cout << "this->DIMS() = "<< derived::DIMS() << " this->size() = " << as_derived().size() << " this_dims "; as_derived().print_dimensions();
					std::cout << "this->DIMS() = "<< deriv::DIMS()   << " param.size() = " << tensor.as_derived().size() << " param_dims "; tensor.as_derived().print_dimensions();
					std::cout << "\n";
					throw std::invalid_argument("Tensor by Tensor operation - size mismatch - ");
				}
	}

	//assert same math library // asserts both memory is allocated on gpu or cpu
	template<class deriv>
	void assert_same_ml(const Tensor_Operations<deriv>& tensor) const {
		static_assert(MTF::same<_mathlib<derived>, _mathlib<deriv>>::conditional, "mathlib_type must be identical");
	}

	//-------------tensor alias -----------------//

	struct tensor_alias {

		Tensor_Operations<derived>& alias;

		tensor_alias(Tensor_Operations<derived>& alias_) : alias(alias_) {}

		template<class tensor>
			derived& operator =(const tensor& param) {
				assert_same_size(param);
				alias.evaluate(alias.bi_expr<oper::alias_assign>(param));
				return alias.as_derived();
			}
			template<class tensor>
			derived& operator +=(const tensor& param) {
				assert_same_size(param);
				alias.evaluate(alias.bi_expr<oper::alias_add_assign>(param));
				return alias.as_derived();
			}
			template<class tensor>
			derived& operator -=(const tensor& param) {
				assert_same_size(param);
				alias.evaluate(alias.bi_expr<oper::alias_sub_assign>(param));
				return alias.as_derived();
			}

			template<class pDeriv>
			derived& operator =(const unsafe_AAD<pDeriv>& param) {
				assert_same_size(param);
				alias.evaluate<false>(alias.bi_expr<oper::alias_assign>(param));
				return alias.as_derived();
			}


			template<class pDeriv>
			derived& operator +=(const unsafe_AAD<pDeriv>& param) {
				assert_same_size(param);
				alias.evaluate<false>(alias.bi_expr<oper::alias_add_assign>(param));
				return alias.as_derived();

			}
			template<class pDeriv>
			derived& operator -=(const unsafe_AAD<pDeriv>& param) {
				assert_same_size(param);
				alias.alias_evaluate<false>(alias.bi_expr<oper::alias_sub_assign>(param));
				return alias.as_derived();
			}
	};

};
}
}
#endif /* TENSOR_CORE_H_ */
