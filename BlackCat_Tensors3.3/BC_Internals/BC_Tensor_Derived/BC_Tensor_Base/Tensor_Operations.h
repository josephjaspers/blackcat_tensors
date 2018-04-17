/*
 * Tensor_Core.h
 *
 *  Created on: Dec 30, 2017
 *      Author: joseph
 */

#ifndef TENSOR_HEAD_H_
#define TENSOR_HEAD_H_

#include "BC_Tensor_Types/Expression_Binary_Pointwise.h"
#include "BC_Tensor_Types/Expression_Binary_Pointwise_ScalarMul.h"
#include "BC_Tensor_Types/Expression_Binary_Functors.h"
#include "BC_Tensor_Types/Expression_Unary_Pointwise.h"
#include "BC_Tensor_Types/Expression_Binary_Dotproduct.h"
#include "BC_Tensor_Types/Expression_Unary_MatrixTransposition.h"
#include "BC_Tensor_Types/Expression_Unary_MaxPooling.h"
#include "BC_Tensor_Types/Expression_Binary_Correlation.h"
#include "BC_Tensor_Types/Expression_Binary_Correlation_Padded.h"

#include <type_traits>
namespace BC {

template<class> struct Tensor_Operations;

template<class A>
struct alternate_asterix_denoter {
	//This class is returned from the overloaded unary (*) operator, we use it to create a secondary subset of operators IE **, %*
	const Tensor_Operations<A>& ref;
	const Tensor_Operations<A>& operator() () const { return ref; }
	const Tensor_Operations<A>& get () const { return ref; }

	alternate_asterix_denoter(const Tensor_Operations<A>& r) : ref(const_cast<Tensor_Operations<A>&>(r)) {}
};

/*
 * This is where the beautiful lazy expressions are created
 */

template<class derived>
struct Tensor_Operations {

	using evaluation_type 	= _evaluation<derived>;
	using functor_type 		= _functor<derived>;
	using scalar_type 		= _scalar<derived>;
	using math_library 		= _mathlib<derived>;
	using this_type = derived;
	template<class shell, class... T> using expr_sub = typename MTF::shell_of<shell>::template type<T...>;


	//determines the return type of pointwise operations
	template<class param_deriv, class functor>
	struct impl {
		using greater_rank_type = std::conditional_t<(derived::DIMS() > param_deriv::DIMS()), derived, param_deriv>;
		using param_functor_type = _functor<param_deriv>;
		using type = 	   expr_sub<greater_rank_type, binary_expression<functor_type ,param_functor_type, functor>, math_library>;
		using unary_type = expr_sub<greater_rank_type, unary_expression <functor_type, functor>, math_library>;
	};

	//determinates the return type of dot-product operations (and scalar multiplication)
	template<class param_deriv>
	struct dp_impl {
		static constexpr bool SCALAR_MUL = derived::DIMS() == 0 || param_deriv::DIMS() == 0;
		using param_functor_type 	= typename Tensor_Operations<param_deriv>::functor_type;
		using greater_rank_type 	= std::conditional_t<(derived::DIMS() > param_deriv::DIMS()), derived, param_deriv>;
		using lesser_rank_type 		= std::conditional_t<(derived::DIMS() < param_deriv::DIMS()), derived, param_deriv>;

		using dot_type 				= binary_expression_dotproduct<_functor<derived>, _functor<param_deriv>, math_library>;
		using scalmul_type 			= binary_expression_scalar_mul< functor_type , param_functor_type>;

		using type = std::conditional_t<!SCALAR_MUL,
						expr_sub<lesser_rank_type, dot_type, math_library>,
						expr_sub<greater_rank_type, scalmul_type, math_library>>;
	};

	//assert either scalar by tensor operation or same size (not same dimensions)
	template<class deriv>
	void  assert_same_size(const Tensor_Operations<deriv>& tensor) const {
#ifdef	BLACKCAT_TENSORS_ASSERT_VALID

		if (derived::DIMS() != 0 && deriv::DIMS() != 0)
		if ((asBase().size() != tensor.asBase().size()) && (this->asBase().DIMS() != 0 && tensor.asBase().DIMS() != 0)) {
			std::cout << "this->DIMS() = "<< derived::DIMS() << " this->size() = " << asBase().size() << " this_dims "; asBase().printDimensions();
			std::cout << "this->DIMS() = "<< deriv::DIMS()   << " param.size() = " << tensor.asBase().size() << " param_dims "; tensor.asBase().printDimensions();
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
		static_assert(MTF::same<_mathlib<derived>, _mathlib<deriv>>::conditional, "math_library must be identical");
#endif
	}

	//Returns the class returned as its most derived member
	 const derived& asBase() const { return static_cast<const derived&>(*this); }
	 	   derived& asBase() 	  { return static_cast<	     derived&>(*this); }
	//Return expression or array of Tensor (both support iterating with bracket operator [])
		    const auto& data() const { return static_cast<const derived&>(*this)._data(); }
		    auto& data()		 	 { return static_cast<		derived&>(*this)._data(); }

	//-------------------------------------dotproduct-----------------------------------------//

	template<class pDeriv>
	auto operator *(const Tensor_Operations<pDeriv>& param) const {
		 return typename dp_impl<pDeriv>::type(this->data(), param.data());
	}
	//--------------------------------------pointwise operators-------------------------------//
	template<class pDeriv>
	typename impl<pDeriv, add>::type operator +(const Tensor_Operations<pDeriv>& param) const {
		assert_same_size(param);
		return typename impl<pDeriv, add>::type(this->data(), param.data());
	}
	template<class pDeriv>
	typename impl<pDeriv, sub>::type operator -(const Tensor_Operations<pDeriv>& param) const {
		assert_same_size(param);
		return typename impl<pDeriv, sub>::type(this->data(), param.data());
	}
	template<class pDeriv>
	typename impl<pDeriv, div>::type operator /(const Tensor_Operations<pDeriv>& param) const {
		assert_same_size(param);
		return typename impl<pDeriv, div>::type(this->data(), param.data());
	}
	template<class pDeriv>
	typename impl<pDeriv, mul>::type operator %(const Tensor_Operations<pDeriv>& param) const {			//overloaded for pointwise multiply
		assert_same_size(param);
		return typename impl<pDeriv, mul>::type(this->data(), param.data());
	}
	template<class pDeriv>
	typename impl<pDeriv, mul>::type operator *(const alternate_asterix_denoter<pDeriv>& param) const { //alternative for pointwise multiply
		assert_same_size(param.get());
		return typename impl<pDeriv, mul>::type(this->data(), param.get().data());
	}
	//--------------------------------------assignment operators-----------------------------------------------//

	//dimensional copy (this is more optimized for certain functions)
	struct copy1d { template<class... params> static void foo(params&... p) { math_library::copy1d(p...); } };
	struct copy2d { template<class... params> static void foo(params&... p) { math_library::copy2d(p...); } };
	struct copy3d { template<class... params> static void foo(params&... p) { math_library::copy3d(p...); } };
	struct copy4d { template<class... params> static void foo(params&... p) { math_library::copy4d(p...); } };

	template<class pDeriv>
	using copy_func =			//this defaults to copy1d, as this library all tensors can utilized 1d copying, though it maybe slower
			std::conditional_t<_functor<pDeriv>::CONTINUOUS() == 0 && functor_type::CONTINUOUS() == 0, copy1d,
				std::conditional_t<_functor<pDeriv>::CONTINUOUS() == 2 && functor_type::CONTINUOUS() == 2, copy2d,
					std::conditional_t<_functor<pDeriv>::CONTINUOUS() == 3 && functor_type::CONTINUOUS() == 3, copy3d,
						std::conditional_t<_functor<pDeriv>::CONTINUOUS() == 4 && functor_type::CONTINUOUS() == 4, copy4d, copy1d>
					>
				>
			>;

	template<class pDeriv>
	derived& operator =(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		if (_functor<pDeriv>::CONTINUOUS() == 0 && _functor<derived>::CONTINUOUS() == 0)
			math_library::copy(asBase().data(), param.asBase().data(), this->asBase().size());
		else {
			copy_func<pDeriv>::foo(asBase().data(), param.asBase().data());
			std::cout << "CHOOSING DIMENSIONAL COPY" << std::endl;
		}
		return static_cast<derived&>(*this);
	}

	template<class pDeriv>
	derived& operator +=(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		return *this = typename impl<pDeriv, add>::type(this->data(), param.data());
	}
	template<class pDeriv>
	derived& operator -=(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		return *this = typename impl<pDeriv, sub>::type(this->data(), param.data());
	}
	template<class pDeriv>
	derived& operator /=(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		return *this = typename impl<pDeriv, div>::type(this->data(), param.data());
	}
	template<class pDeriv>
	derived& operator %=(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		return *this = typename impl<pDeriv, mul>::type(this->data(), param.data());
	}


	//-------------------------------------------DELAYED ASSIGNMENT OPERATORS---------------------------------------------//
	template<class pDeriv>
	typename impl<pDeriv, assign>::type operator ==(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		return typename impl<pDeriv, assign>::type(this->data(), param.data());
	}
	template<class pDeriv>
	typename impl<pDeriv, assign>::type operator =(const alternate_asterix_denoter<pDeriv>& param) {
		assert_same_size(param);
		return typename impl<pDeriv, assign>::type(this->data(), param.data());
	}
	template<class pDeriv>
	typename impl<pDeriv, mul>::type operator +=(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return typename impl<pDeriv, add>::type(this->data(), param.get().data());
	}
	template<class pDeriv>
	typename impl<pDeriv, mul>::type operator -=(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return typename impl<pDeriv, sub>::type(this->data(), param.get().data());
	}
	template<class pDeriv>
	typename impl<pDeriv, mul>::type operator %=(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return typename impl<pDeriv, mul>::type(this->data(), param.get().data());
	}
	template<class pDeriv>
	typename impl<pDeriv, mul>::type operator *=(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return typename impl<pDeriv, mul>::type(this->data(), param.get().data());
	}
	template<class pDeriv>
	typename impl<pDeriv, div>::type operator /=(const alternate_asterix_denoter<pDeriv>& param) const {
		assert_same_size(param.get());
		return typename impl<pDeriv, add>::type(this->data(), param.get().data());
	}

	//-----------------------------------COMBINE EXPRESSION-------------------------------------------------//
	template<class pDeriv>
	typename impl<pDeriv, combine>::type operator &&(const Tensor_Operations<pDeriv>& param) {
		assert_same_size(param);
		return typename impl<pDeriv, combine>::type(this->data(), param.data());
	}

	//-----------------------------------custom expressions--------------------------------------------------//
	template<class functor>
	auto unExpr(functor f) const {
		return typename impl<derived, functor>::unary_type(asBase().data(), f);
	}
	template<class d2, class functor>
	auto binExpr(functor f, const Tensor_Operations<d2>& rv) {
			assert_same_size(rv);
		return typename impl<d2, functor>::type(asBase().data(), rv.asBase().data());
	}
	template<class functor>
	auto unExpr() const {
		return typename impl<derived, functor>::unary_type(asBase().data());
	}
	template<class functor, class d2>
	auto binExpr(const Tensor_Operations<d2>& rv) {
		return typename impl<d2, functor>::type(asBase().data(), rv.asBase().data());
	}
	//------------------------------------alternate asterix denoter----------------------------------//
	 const alternate_asterix_denoter<derived> operator * () const {
		return alternate_asterix_denoter<derived>(*this);
	}

	 //--------------------------------HIGHER ORDER OPERATIONS------------------------------//
		template<class deriv>
		auto corr(const Tensor_Operations<deriv>& rv) {
			assert_same_size(rv);
			return
						typename base<0>::template type<
							binary_expression_correlation<functor_type, _functor<deriv>, 0>, math_library>
			(asBase().data(), rv.asBase().data());
		}

	template<int mv = 2, class deriv>
	auto x_corr(const Tensor_Operations<deriv>& rv) {

		return
				typename base<mv>::template type<
					binary_expression_correlation<functor_type,  _functor<deriv>, mv>, math_library
				>(asBase().data(), rv.asBase().data());
	}

	template<int mv = 2, class deriv>
	auto x_corr_padded(const Tensor_Operations<deriv>& rv) {

		return
				typename base<mv>::template type<
				binary_expression_correlation_padded<functor_type, _functor<deriv>, mv>, math_library>
					(asBase().data(), rv.asBase().data());
	}
	template<int search_space = 3>
	auto max_pooling() {
		return expr_sub<derived, unary_expression_maxpooling<functor_type, search_space>, math_library>(asBase().data());
	}

};
}
#endif /* TENSOR_CORE_H_ */
