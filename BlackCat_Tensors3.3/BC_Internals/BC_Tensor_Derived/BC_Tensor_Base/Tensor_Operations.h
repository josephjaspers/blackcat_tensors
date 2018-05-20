/*
 * Core.h
 *
 *  Created on: Dec 30, 2017
 *      Author: joseph
 */

#ifndef TENSOR_HEAD_H_
#define TENSOR_HEAD_H_

#include "BC_Tensor_Types/Expression_Binary_Pointwise.h"
#include "BC_Tensor_Types/Expression_Binary_Functors.h"
#include "BC_Tensor_Types/Expression_Binary_Dotproduct.h"

#include "BC_Tensor_Types/Expression_Unary_Pointwise.h"
#include "BC_Tensor_Types/Expression_Unary_Functors.h"

#include "BC_Tensor_Types/Expression_Unary_MatrixTransposition.h"
#include "BC_Tensor_Types/Expression_Unary_MaxPooling.h"
#include "BC_Tensor_Types/Expression_Unary_Cacher.h"
#include "BC_Tensor_Types/Expression_Binary_Correlation.h"
#include "BC_Tensor_Types/Expression_Binary_Correlation_Padded.h"
#include "BC_Tensor_Types/Expression_Binary_Correlation_Stack.h"
#include "BC_Tensor_Types/Expression_Binary_Correlation_Padded_Stack.h"

#include "BC_Tensor_Types/Expression_Binary_Correlation_Error.h"


#include "Operations_Utility/AlternateAsterixDenoter.h"
#include "Operations_Utility/Expression_Determiner.h"
#include "Operations_Utility/Unary_Functions.h"
#include <type_traits>

namespace BC {
 //This is where the beautiful lazy expressions are created

template<class derived>
struct Tensor_Operations {

	template<class> friend class Tensor_Operations;
	template<class pderiv, class functor> using impl 	= typename expression_determiner<derived>::template impl<pderiv, functor>;
	template<class pderiv> 				  using dp_impl	= typename expression_determiner<derived>::template dp_impl<pderiv>;

	using functor_type 		= _functor<derived>;
	using scalar_type 		= _scalar<derived>;
	using mathlib_type 		= _mathlib<derived>;

private:
	//Returns the class returned as its most derived member
	 const derived& as_derived() const { return static_cast<const derived&>(*this);  }
	 	   derived& as_derived() 	  { return static_cast<	     derived&>(*this); }
public:

	void randomize(scalar_type lb, scalar_type ub)  { mathlib_type::randomize(as_derived().data(), lb, ub); }
	void fill(scalar_type value) 					{ mathlib_type::fill(as_derived().data(), value); }
	void zero() 									{ mathlib_type::zero(as_derived().data()); }

	//-------------------------------------dotproduct-------------------- ---------------------//

	template<class pDeriv>
	auto operator *(const Tensor_Operations<pDeriv>& param) const {
		 return typename dp_impl<pDeriv>::type(as_derived().data(), param.as_derived().data());
	}
	//--------------------------------------pointwise operators-------------------------------//
	template<class pDeriv>
	typename impl<pDeriv, add>::type operator +(const Tensor_Operations<pDeriv>& param) const {
		assert_same_size(param);
		return bi_expr<add>(param);
	}
	template<class pDeriv>
	typename impl<pDeriv, sub>::type operator -(const Tensor_Operations<pDeriv>& param) const {
		assert_same_size(param);
		return bi_expr<sub>(param);
	}
	template<class pDeriv>
	typename impl<pDeriv, div>::type operator /(const Tensor_Operations<pDeriv>& param) const {
		assert_same_size(param);
		return bi_expr<div>(param);
	}
	template<class pDeriv>
	typename impl<pDeriv, mul>::type operator %(const Tensor_Operations<pDeriv>& param) const {			//overloaded for pointwise multiply
		assert_same_size(param);
		return bi_expr<mul>(param);
	}
	template<class pDeriv>
	typename impl<pDeriv, mul>::type operator *(const alternate_asterix_denoter<pDeriv>& param) const { //alternative for pointwise multiply
		assert_same_size(param.get());
		return bi_expr<mul>(param.get());
	}
	//--------------------------------------assignment operators-----------------------------------------------//
private:
	template<class t>
	static void evaluate(const Tensor_Operations<t>& tensor) {
		static constexpr int iterator_dimension = _functor<t>::CONTINUOUS();
		mathlib_type::template dimension<iterator_dimension>::eval(tensor.as_derived().data());
	}
public:

	template<class pDeriv>
	derived& operator =(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		evaluate(bi_expr<assign>(param));
		return as_derived();
	}

	template<class pDeriv>
	derived& operator +=(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		evaluate(bi_expr<add_assign>(param));
		return as_derived();

	}
	template<class pDeriv>
	derived& operator -=(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		evaluate(bi_expr<sub_assign>(param));
		return as_derived();
	}
	template<class pDeriv>
	derived& operator /=(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		evaluate(bi_expr<div_assign>(param));
		return as_derived();
	}
	template<class pDeriv>
	derived& operator %=(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		evaluate(bi_expr<mul_assign>(param));
		return as_derived();
	}
	 //--------------------------------Other Operators------------------------------//

	 auto operator - () const {
		 return this->un_expr<oper::negation>();
	 }
	template<class pDeriv>
	auto operator ==(const Tensor_Operations<pDeriv>& param) const {
		assert_same_size(param);
		return bi_expr<equal>(param);
	}
	template<class pDeriv>
	auto operator >(const Tensor_Operations<pDeriv>& param) const {
		assert_same_size(param);
		return bi_expr<greater>(param);
	}
	template<class pDeriv>
	auto operator <(const Tensor_Operations<pDeriv>& param) const {
		assert_same_size(param);
		return bi_expr<lesser>(param);
	}
	template<class pDeriv>
	auto operator >=(const Tensor_Operations<pDeriv>& param) const {
		assert_same_size(param);
		return bi_expr<greater_equal>(param);
	}
	template<class pDeriv>
	auto operator <=(const Tensor_Operations<pDeriv>& param) const {
		assert_same_size(param);
		return bi_expr<lesser_equal>(param);
	}

	//-------------------------------------------DELAYED ASSIGNMENT OPERATORS---------------------------------------------//
	template<class pDeriv>
	auto operator =(const alternate_asterix_denoter<pDeriv>& param) {
		assert_same_size(param);
		return bi_expr<_cache>(param);
	}

	template<class pDeriv>
	auto operator +=(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return bi_expr<add_assign>(param.get());
	}
	template<class pDeriv>
	auto operator -=(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return bi_expr<sub_assign>(param.get());
	}
	template<class pDeriv>
	auto operator %=(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return bi_expr<mul_assign>(param.get());
	}
	template<class pDeriv>
	auto operator *=(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return bi_expr<mul_assign>(param.get());
	}
	template<class pDeriv>
	auto operator /=(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return bi_expr<div_assign>(param.get());
	}

	//-----------------------------------COMBINE EXPRESSION-------------------------------------------------//
	template<class pDeriv>
	typename impl<pDeriv, combine>::type operator &&(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		return bi_expr<combine>(param);
	}


	//-----------------------------------custom expressions--------------------------------------------------//
	template<class functor>
	auto un_expr(functor f) const {
		return typename impl<derived, functor>::unary_type(as_derived().data(), f);
	}
	template<class functor>
	const auto un_expr() const {
		return typename impl<derived, functor>::unary_type(as_derived().data());
	}
	template<class d2, class functor>
	const auto bi_expr(functor f, const Tensor_Operations<d2>& rv) const {
		return typename impl<d2, functor>::type(as_derived().data(), rv.as_derived().data());
	}
	template<class functor, class d2>
	const auto bi_expr(const Tensor_Operations<d2>& rv) const {
		return typename impl<d2, functor>::type(as_derived().data(), rv.as_derived().data());
	}
	//------------------------------------alternate asterix denoter----------------------------------//
	 const alternate_asterix_denoter<derived> operator * () const {
		return alternate_asterix_denoter<derived>(*this);
	}
	 //--------------------------------More_Complex Operations------------------------------//
	template<class deriv>
	auto corr(const Tensor_Operations<deriv>& rv) const {
		assert_same_size(rv);
		return typename tensor_of<0>::template type<
				binary_expression<functor_type, _functor<deriv>,
						_x_corr<1, inner>>, mathlib_type>(as_derived().data(),
				rv.as_derived().data());
	}

	template<int mv, class type = inner, class deriv>
	auto x_corr(const Tensor_Operations<deriv>& rv) const {

		return typename tensor_of<mv>::template type<
				binary_expression<functor_type, _functor<deriv>,
						_x_corr<mv, type>>, mathlib_type>(as_derived().data(),
				rv.as_derived().data());
	}
	template<int mv, class type = inner, class deriv>
	auto x_corr_stack(const Tensor_Operations<deriv>& rv) const {

		return typename tensor_of<mv + 1>::template type<
				binary_expression<functor_type, _functor<deriv>,
						_x_corr_stack<mv, type>>, mathlib_type>(
				as_derived().data(), rv.as_derived().data());
	}



	 //--------------------------------ASSERTIONS------------------------------//


	//assert either scalar by tensor operation or same size (not same dimensions)
	template<class deriv>
	void  assert_same_size(const Tensor_Operations<deriv>& tensor) const {
#ifndef	BC_RELEASE

		if (derived::DIMS() != 0 && deriv::DIMS() != 0)
		if ((as_derived().size() != tensor.as_derived().size()) && (this->as_derived().DIMS() != 0 && tensor.as_derived().DIMS() != 0)) {
			std::cout << "this->DIMS() = "<< derived::DIMS() << " this->size() = " << as_derived().size() << " this_dims "; as_derived().printDimensions();
			std::cout << "this->DIMS() = "<< deriv::DIMS()   << " param.size() = " << tensor.as_derived().size() << " param_dims "; tensor.as_derived().printDimensions();
			std::cout << "\n";
			throw std::invalid_argument("Tensor by Tensor operation - size mismatch - ");
		}
		assert_same_ml(tensor);
#endif
	}

	//assert same math library // asserts both memory is allocated on gpu or cpu
	template<class deriv>
	void assert_same_ml(const Tensor_Operations<deriv>& tensor) const {
#ifndef BC_RELEASE
		static_assert(MTF::same<_mathlib<derived>, _mathlib<deriv>>::conditional, "mathlib_type must be identical");
#endif
	}

};
}
#endif /* TENSOR_CORE_H_ */
