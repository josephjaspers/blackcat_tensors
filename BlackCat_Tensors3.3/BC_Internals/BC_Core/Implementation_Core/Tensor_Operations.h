/*
 * Tensor_Core.h
 *
 *  Created on: Dec 30, 2017
 *      Author: joseph
 */

#ifndef TENSOR_HEAD_H_
#define TENSOR_HEAD_H_

#include "../../BC_MetaTemplateFunctions/Simple.h"
#include "../../BC_MetaTemplateFunctions/Adhoc.h"
#include "BC_Expressions/Expression_Binary_Pointwise.h"
#include "BC_Expressions/Expression_Unary_Pointwise.h"
#include "BC_Expressions/Expression_Binary_Dotproduct.h"
#include "BC_Expressions/Expression_Unary_MatrixTransposition.h"
#include "BC_Expressions/Expression_Binary_Correlation.h"

#include "Determiners.h"
#include <type_traits>
namespace BC {

template<class> struct Tensor_Operations;

template<class A>
struct alternate_asterix_denoter {
	//This class is returned from the overloaded unary (*) operator, we use it to create a secondary subset of operators IE **, %*
	const Tensor_Operations<A>& ref;
	const Tensor_Operations<A>& operator() () const { return ref; }
	const Tensor_Operations<A>& get () const { return ref; }

	alternate_asterix_denoter(Tensor_Operations<A>& r) : ref(r) {}
};

/*
 *
 * This is where the beautiful lazy expressions are created
 *
 */

template<class derived>
struct Tensor_Operations {

	using evaluation_type 	= _evaluation<derived>;
	using functor_type 		= _functor<derived>;
	using scalar_type 		= _scalar<derived>;
	using math_library 		= _mathlib<derived>;

	using this_type = derived;

	template<class param_deriv, class functor>
	struct impl {
		using greater_rank_type = std::conditional_t<(derived::RANK() > param_deriv::RANK()), derived, param_deriv>;
		using param_functor_type = typename Tensor_Operations<param_deriv>::functor_type;
		using type = 	   typename MTF::expression_substitution<binary_expression<scalar_type, functor, functor_type ,param_functor_type>, greater_rank_type>::type;
		using unary_type = typename MTF::expression_substitution<unary_expression <scalar_type, functor, functor_type>, 					greater_rank_type>::type;
	};
	template<class param_deriv>
	struct dp_impl {
		static constexpr bool SCALAR_MUL = derived::RANK() == 0 || param_deriv::RANK() == 0;
		using param_functor_type 	= typename Tensor_Operations<param_deriv>::functor_type;
		using greater_rank_type 	= std::conditional_t<(derived::RANK() > param_deriv::RANK()), derived, param_deriv>;
		using lesser_rank_type 		= std::conditional_t<(derived::RANK() < param_deriv::RANK()), derived, param_deriv>;

		using dot_type 				= binary_expression_dotproduct<scalar_type, _functor<derived>, _functor<param_deriv>, math_library>;
		using scalmul_type 			= binary_expression_scalar_mul<scalar_type, functor_type ,param_functor_type>;

		using type = std::conditional_t<!SCALAR_MUL,
						typename MTF::expression_substitution<dot_type, lesser_rank_type>::type,
						typename MTF::expression_substitution<scalmul_type, greater_rank_type>::type>;
	};


	template<class deriv>  __BCinline__
	void  assert_same_size(const Tensor_Operations<deriv>& tensor) const {
#ifdef	BLACKCAT_TENSORS_ASSERT_VALID

		if (derived::RANK() != 0 && deriv::RANK() != 0)
		if ((asBase().size() != tensor.asBase().size()) && (this->asBase().RANK() != 0 && tensor.asBase().RANK() != 0)) {
			std::cout << "this->RANK() = "<< derived::RANK() << " this->size() = " << asBase().size() << " this_dims "; asBase().printDimensions();
			std::cout << "this->RANK() = "<< deriv::RANK()   << " param.size() = " << tensor.asBase().size() << " param_dims "; tensor.asBase().printDimensions();
			std::cout << "\n";
			throw std::invalid_argument("Tensor by Tensor operation - size mismatch - ");
		}
		assert_same_ml(tensor);
#endif
	}

	template<class deriv> __BCinline__
	void assert_same_ml(const Tensor_Operations<deriv>& tensor) const {
#ifdef BLACKCAT_TENSORS_ASSERT_VALID
		static_assert(MTF::same<math_library, typename Tensor_Operations<deriv>::math_library>::conditional, "math_library must be identical");
#endif
	}

	//Returns the class returned as its most derived member
	__BCinline__ const derived& asBase() const { return static_cast<const derived&>(*this); }
	__BCinline__ 	   derived& asBase() 	  { return static_cast<	     derived&>(*this); }
	//Return expression or array of Tensor (both support iterating with bracket operator [])
		  __BCinline__  const auto& data() const { return static_cast<const derived&>(*this)._data(); }
		  __BCinline__  auto& data()		 { return static_cast<		derived&>(*this)._data(); }

	//-------------------------------------dotproduct-----------------------------------------//

	template<class pDeriv> __BCinline__
	typename dp_impl<pDeriv>::type operator *(const Tensor_Operations<pDeriv>& param) const {
		 return typename dp_impl<pDeriv>::type(this->data(), param.data());
	}
	//--------------------------------------pointwise operators-------------------------------//
	template<class pDeriv> __BCinline__
	typename impl<pDeriv, add>::type operator +(const Tensor_Operations<pDeriv>& param) const {
		assert_same_size(param);
		return typename impl<pDeriv, add>::type(this->data(), param.data());
	}
	template<class pDeriv> __BCinline__
	typename impl<pDeriv, sub>::type operator -(const Tensor_Operations<pDeriv>& param) const {
		assert_same_size(param);
		return typename impl<pDeriv, sub>::type(this->data(), param.data());
	}
	template<class pDeriv> __BCinline__
	typename impl<pDeriv, div>::type operator /(const Tensor_Operations<pDeriv>& param) const {
		assert_same_size(param);
		return typename impl<pDeriv, div>::type(this->data(), param.data());
	}
	template<class pDeriv> __BCinline__
	typename impl<pDeriv, mul>::type operator %(const Tensor_Operations<pDeriv>& param) const {			//overloaded for pointwise multiply
		assert_same_size(param);
		return typename impl<pDeriv, mul>::type(this->data(), param.data());
	}
	template<class pDeriv> __BCinline__
	typename impl<pDeriv, mul>::type operator *(const alternate_asterix_denoter<pDeriv>& param) const { //alternative for pointwise multiply
		assert_same_size(param.get());
		return typename impl<pDeriv, mul>::type(this->data(), param.get().data());
	}
	//--------------------------------------assignment operators-----------------------------------------------//
	template<class pDeriv> __BCinline__
	derived& operator =(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		math_library::copy(asBase().data(), param.asBase().data(), this->asBase().size());
		return static_cast<derived&>(*this);
	}
	template<class pDeriv> __BCinline__
	derived& operator +=(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		return *this = typename impl<pDeriv, add>::type(this->data(), param.data());
	}
	template<class pDeriv> __BCinline__
	derived& operator -=(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		return *this = typename impl<pDeriv, sub>::type(this->data(), param.data());
	}
	template<class pDeriv> __BCinline__
	derived& operator /=(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		return *this = typename impl<pDeriv, div>::type(this->data(), param.data());
	}
	template<class pDeriv> __BCinline__
	derived& operator %=(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		return *this = typename impl<pDeriv, mul>::type(this->data(), param.data());
	}


	//-------------------------------------------DELAYED ASSIGNMENT OPERATORS---------------------------------------------//
	template<class pDeriv> __BCinline__
	typename impl<pDeriv, assign>::type operator ==(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		return typename impl<pDeriv, assign>::type(this->data(), param.data());
	}
	template<class pDeriv> __BCinline__
	typename impl<pDeriv, assign>::type operator =(const alternate_asterix_denoter<pDeriv>& param) {
		assert_same_size(param);
		return typename impl<pDeriv, assign>::type(this->data(), param.data());
	}
	template<class pDeriv> __BCinline__
	typename impl<pDeriv, mul>::type operator +=(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return typename impl<pDeriv, add>::type(this->data(), param.get().data());
	}
	template<class pDeriv> __BCinline__
	typename impl<pDeriv, mul>::type operator -=(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return typename impl<pDeriv, sub>::type(this->data(), param.get().data());
	}
	template<class pDeriv> __BCinline__
	typename impl<pDeriv, mul>::type operator %=(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return typename impl<pDeriv, mul>::type(this->data(), param.get().data());
	}
	template<class pDeriv> __BCinline__
	typename impl<pDeriv, div>::type operator /=(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return typename impl<pDeriv, add>::type(this->data(), param.get().data());
	}

	//-----------------------------------COMBINE EXPRESSION-------------------------------------------------//
	template<class pDeriv> __BCinline__
	typename impl<pDeriv, combine>::type operator &&(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		return typename impl<pDeriv, combine>::type(this->data(), param.data());
	}

	//-----------------------------------custom expressions--------------------------------------------------//
	template<class functor> __BCinline__
	auto unExpr(functor f) const {
		return typename impl<derived, functor>::unary_type(asBase().data());
	}
	template<class d2, class functor> __BCinline__
	auto binExpr(functor f, const Tensor_Operations<d2>& rv) {
			assert_same_size(rv);
		return typename impl<d2, functor>::type(asBase().data(), rv.asBase().data());
	}
	template<class functor> __BCinline__
	auto unExpr() const {
		return typename impl<derived, functor>::unary_type(asBase().data());
	}
	template<class functor, class d2> __BCinline__
	auto binExpr(const Tensor_Operations<d2>& rv) {
		return typename impl<d2, functor>::type(asBase().data(), rv.asBase().data());
	}
	//------------------------------------alternate asterix denoter----------------------------------//
	__BCinline__ const alternate_asterix_denoter<derived> operator * () const {
		return alternate_asterix_denoter<derived>(this);
	}

//	------------------------------------higher order functions----------------------------------//
	template<class pDeriv> __BCinline__
	auto x_corr(const Tensor_Operations<pDeriv>& param) const {
		assert_same_ml(param);
		return typename MTF::expression_substitution<
					binary_expression_correlation<scalar_type, functor_type ,_functor<pDeriv>>, derived
						>::type(this->data(), param.data());
	}
};
}
#endif /* TENSOR_CORE_H_ */
