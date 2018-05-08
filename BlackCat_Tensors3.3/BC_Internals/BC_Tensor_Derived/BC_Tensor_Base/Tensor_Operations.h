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

#include "Operations_Utility/AlternateAsterixDenoter.h"
#include "Operations_Utility/Expression_Determiner.h"
#include <type_traits>

namespace BC {
 //This is where the beautiful lazy expressions are created

template<class derived>
struct Tensor_Operations {

	template<class> friend class Tensor_Operations;
	template<class pderiv, class functor> using impl 	= typename expression_determiner<derived>::template impl<pderiv, functor>;
	template<class pderiv> 				  using dp_impl	= typename expression_determiner<derived>::template dp_impl<pderiv>;

	using evaluation_type 	= _evaluation<derived>;
	using functor_type 		= _functor<derived>;
	using scalar_type 		= _scalar<derived>;
	using mathlib_type 		= _mathlib<derived>;

private:
	//Returns the class returned as its most derived member
	 const derived& asDerived() const { return static_cast<const derived&>(*this);  }
	 	   derived& asDerived() 	  { return static_cast<	     derived&>(*this); }
public:

	void randomize(scalar_type lb, scalar_type ub) { mathlib_type::randomize(asDerived().data(), lb, ub, asDerived().size()); }
	void fill(scalar_type value) { mathlib_type::fill(asDerived().data(), value, asDerived().size()); }
	void zero() { mathlib_type::zero(asDerived().data(), asDerived().size()); }

	//-------------------------------------dotproduct-------------------- ---------------------//

	template<class pDeriv>
	auto operator *(const Tensor_Operations<pDeriv>& param) const {
		 return typename dp_impl<pDeriv>::type(asDerived().data(), param.asDerived().data());
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

	//non continuous copy
	template<class pDeriv>
	std::enable_if_t<(_functor<pDeriv>::CONTINUOUS() != 0 || _functor<derived>::CONTINUOUS() != 0), derived&>
	operator =(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		static constexpr int iter_dimension = max(_functor<pDeriv>::CONTINUOUS(), _functor<derived>::CONTINUOUS());
		mathlib_type::template dimension<iter_dimension>::copy(asDerived().data(), param.asDerived().data());

		return static_cast<derived&>(*this);
	}
	//continuous copy
	template<class pDeriv>
	std::enable_if_t<(_functor<pDeriv>::CONTINUOUS() == 0 && _functor<derived>::CONTINUOUS() == 0), derived&>
		operator =(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		mathlib_type::copy(asDerived().data(), param.asDerived().data(), this->asDerived().size());
		return static_cast<derived&>(*this);
	}

	template<class pDeriv>
	derived& operator +=(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		return *this = bi_expr<add>(param);
	}
	template<class pDeriv>
	derived& operator -=(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		return *this = bi_expr<sub>(param);
	}
	template<class pDeriv>
	derived& operator /=(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		return *this = bi_expr<div>(param);
	}
	template<class pDeriv>
	derived& operator %=(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		return *this = bi_expr<mul>(param);
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
		return bi_expr<add_assign>(param);
	}
	template<class pDeriv>
	auto operator -=(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return bi_expr<sub_assign>(param);
	}
	template<class pDeriv>
	auto operator %=(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return bi_expr<mul_assign>(param);
	}
	template<class pDeriv>
	auto operator *=(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return bi_expr<mul_assign>(param);
	}
	template<class pDeriv>
	auto operator /=(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return bi_expr<div_assign>(param);
	}

	//-----------------------------------COMBINE EXPRESSION-------------------------------------------------//
	template<class pDeriv>
	typename impl<pDeriv, combine>::type operator &&(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		return bi_expr<combine>(param);
	}
	 //--------------------------------Unary_Functions------------------------------//

	 auto operator - () const {
		 return this->un_expr<oper::negation>();
	 }
	 static auto zero(Tensor_Operations& tensor) {
		 return tensor.un_expr<oper::zero>();
	 }
	 static auto one(Tensor_Operations& tensor) {
		 return tensor.un_expr<oper::one>();
	 }
	 static auto fix(Tensor_Operations& tensor) {
		 return tensor.un_expr<oper::fix>();
	 }
	 static auto abs(const Tensor_Operations& tensor) {
		 return tensor.un_expr<oper::abs>();
	 }
	 static auto negation(const Tensor_Operations& tensor) {
		 return tensor.un_expr<oper::negation>();
	 }
	 static auto logical(const Tensor_Operations& tensor) {
		 return tensor.un_expr<oper::logical>();
	 }

	//-----------------------------------custom expressions--------------------------------------------------//
	template<class functor>
	auto un_expr(functor f) const {
		return typename impl<derived, functor>::unary_type(asDerived().data(), f);
	}
	template<class functor>
	const auto un_expr() const {
		return typename impl<derived, functor>::unary_type(asDerived().data());
	}
	template<class d2, class functor>
	const auto bi_expr(functor f, const Tensor_Operations<d2>& rv) const {
			assert_same_size(rv);
		return typename impl<d2, functor>::type(asDerived().data(), rv.asDerived().data());
	}
	template<class functor, class d2>
	const auto bi_expr(const Tensor_Operations<d2>& rv) const {
		return typename impl<d2, functor>::type(asDerived().data(), rv.asDerived().data());
	}
	//------------------------------------alternate asterix denoter----------------------------------//
	 const alternate_asterix_denoter<derived> operator * () const {
		return alternate_asterix_denoter<derived>(*this);
	}
	 //--------------------------------More_Complex Operations------------------------------//
	template<class deriv>
	auto corr(const Tensor_Operations<deriv>& rv) {
		assert_same_size(rv);
		return typename tensor_of<0>::template type<
				binary_expression<functor_type, _functor<deriv>,_x_corr<1, inner>>, mathlib_type>(asDerived().data(),rv.asDerived().data());
	}

	template<int mv, class type = inner, class deriv>
	auto x_corr(const Tensor_Operations<deriv>& rv) {

		return typename tensor_of<mv>::template type<
			binary_expression<functor_type, _functor<deriv>, _x_corr<mv, type>>, mathlib_type>(asDerived().data(),rv.asDerived().data());
	}
	template<int mv, class type = inner, class deriv>
	auto x_corr_stack(const Tensor_Operations<deriv>& rv) {

		return typename tensor_of<mv + 1>::template type<
				binary_expression<functor_type, _functor<deriv>, _x_corr_stack<mv, type>>, mathlib_type>(asDerived().data(),rv.asDerived().data());
	}



	 //--------------------------------ASSERTIONS------------------------------//


	//assert either scalar by tensor operation or same size (not same dimensions)
	template<class deriv>
	void  assert_same_size(const Tensor_Operations<deriv>& tensor) const {
#ifdef	BLACKCAT_TENSORS_ASSERT_VALID

		if (derived::DIMS() != 0 && deriv::DIMS() != 0)
		if ((asDerived().size() != tensor.asDerived().size()) && (this->asDerived().DIMS() != 0 && tensor.asDerived().DIMS() != 0)) {
			std::cout << "this->DIMS() = "<< derived::DIMS() << " this->size() = " << asDerived().size() << " this_dims "; asDerived().printDimensions();
			std::cout << "this->DIMS() = "<< deriv::DIMS()   << " param.size() = " << tensor.asDerived().size() << " param_dims "; tensor.asDerived().printDimensions();
			std::cout << "\n";
			throw std::invalid_argument("Tensor by Tensor operation - size mismatch - ");
		}
		assert_same_ml(tensor);
#endif
	}

	//assert same math library // asserts both memory is allocated on gpu or cpu
	template<class deriv>
	void assert_same_ml(const Tensor_Operations<deriv>& tensor) const {
#ifdef BLACKCAT_TENSORS_ASSERT_VALID
		static_assert(MTF::same<_mathlib<derived>, _mathlib<deriv>>::conditional, "mathlib_type must be identical");
#endif
	}

};
}
#endif /* TENSOR_CORE_H_ */
