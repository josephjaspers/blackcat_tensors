/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef TENSOR_HEAD_H_
#define TENSOR_HEAD_H_

#include "Expression_Templates/Array_Scalar_Constant.h"
#include "Tensor_Common.h"
#include "Expression_Templates/Operations/Binary.h"
#include "Expression_Templates/Operations/Unary.h"
#include "Expression_Templates/Operations/BLAS.h"
#include "Expression_Templates/Expression_Unary.h"
#include "Expression_Templates/Expression_Binary.h"

#include "Expression_Templates/Function_transpose.h"
#include "Expression_Templates/Function_dot.h"
#include "Expression_Templates/Function_ger.h"
#include "Expression_Templates/Function_gemv.h"
#include "Expression_Templates/Function_gemm.h"

#include "Expression_Templates/Tree_Evaluator_Runner.h"

#include "stl_style_Iterators/Coefficientwise_Iterator.h"
#include "stl_style_Iterators/Multidimensional_Iterator.h"


namespace BC {
template<class internal_t> class Tensor_Base;

namespace module {
template<class derived> class Tensor_Operations;
//This is where the beautiful lazy expressions are created

template<class internal_type>
class Tensor_Operations<Tensor_Base<internal_type>> {

	template<class> friend class Tensor_Operations;

	using derived 			= Tensor_Base<internal_type>;
	using internal_t 		= internal_type;
	using scalar_t 			= typename internal_t::scalar_t;
	using allocator_t 		= typename internal_t::allocator_t;

	template<class deriv> using internal_t_of  = typename Tensor_Operations<deriv>::internal_t;
	template<class deriv> using allocator_t_of = typename Tensor_Operations<deriv>::allocator_t;

	static constexpr bool	copy_assignable = internal::BC_array_copy_assignable<internal_type>();
	#define BC_ASSERT_ASSIGNABLE(literal) static_assert(copy_assignable, "ASSERT COPY ASSIGNABLE: " literal)

	template<class expr> 		   using Unary_Expression_t  = BC::Tensor_Base<internal::Unary_Expression<internal_t, expr>>;
	template<class rv, class expr> using Binary_Expression_t = BC::Tensor_Base<internal::Binary_Expression<internal_t, rv, expr>>;

	const derived& as_derived() const { return static_cast<const derived&>(*this); }
	 	  derived& as_derived() 	  { return static_cast<	     derived&>(*this); }

	//--------------------------------------evaluation implementation-----------------------------------------------//
	template<class derived_t>
	void evaluate(const Tensor_Operations<derived_t>& tensor) {
		internal::evaluate(tensor.as_derived().internal());
	}


public:
	//--------------------------------------assignment operators-----------------------------------------------//
#define BC_OPER_ASSIGNMENT_DEF(op, op_functor)													\
																								\
	template<class pDeriv>																		\
	derived& operator op (const Tensor_Operations<pDeriv>& param) {								\
		BC_ASSERT_ASSIGNABLE("derived& operator " #op "(const Tensor_Operations<pDeriv>& param)");	\
		assert_valid(param);																	\
		evaluate(bi_expr< internal::oper:: op_functor >(param));								\
		return as_derived();																	\
	}

	BC_OPER_ASSIGNMENT_DEF(=, assign);
	BC_OPER_ASSIGNMENT_DEF(+=, add_assign);
	BC_OPER_ASSIGNMENT_DEF(-=, sub_assign);
	BC_OPER_ASSIGNMENT_DEF(%=, mul_assign);
	BC_OPER_ASSIGNMENT_DEF(/=, div_assign);

	//--------------------------------------pointwise operators-------------------------------//

#define BC_OPER_COEFFICIENTWISE_DEF(op, op_functor)							\
																			\
	template<class pDeriv>													\
	auto operator op (const Tensor_Operations<pDeriv>& param) const {		\
		assert_valid(param);												\
		return bi_expr< internal::oper:: op_functor >(param);				\
	}

	BC_OPER_COEFFICIENTWISE_DEF(+, add)
	BC_OPER_COEFFICIENTWISE_DEF(-, sub)
	BC_OPER_COEFFICIENTWISE_DEF(%, mul)
	BC_OPER_COEFFICIENTWISE_DEF(/, div)
	BC_OPER_COEFFICIENTWISE_DEF( == , equal )
	BC_OPER_COEFFICIENTWISE_DEF( >  , greater)
	BC_OPER_COEFFICIENTWISE_DEF( <  , lesser)
	BC_OPER_COEFFICIENTWISE_DEF( >= , greater_equal)
	BC_OPER_COEFFICIENTWISE_DEF( <= , lesser_equal )


//----------------------------------------------scalar assignment operations--------------------------------------------------//
	template<class p_scalar_t>
	using enable_if_convertible = std::enable_if_t<std::is_convertible<p_scalar_t, scalar_t>::value>;

#define BC_OPER_SCALAR_ASSIGNMENT_DEF(op, op_functor)																		\
	template<class p_scalar_t, typename = enable_if_convertible<p_scalar_t>>												\
	derived& operator  op (const p_scalar_t& param) {																			\
		BC_ASSERT_ASSIGNABLE("derived& operator " #op " (const Tensor_Operations<pDeriv>& param)");								\
		evaluate(bi_expr_internal<internal::oper:: op_functor >(internal::scalar_constant<allocator_t>((scalar_t)param)));		\
		return as_derived();																								\
	}

	BC_OPER_SCALAR_ASSIGNMENT_DEF(=, assign)
	BC_OPER_SCALAR_ASSIGNMENT_DEF(+=, add_assign)
	BC_OPER_SCALAR_ASSIGNMENT_DEF(-=, sub_assign)
	BC_OPER_SCALAR_ASSIGNMENT_DEF(/=, div_assign)
	BC_OPER_SCALAR_ASSIGNMENT_DEF(%=, mul_assign)

#define BC_OPER_SCALAR_BASIC_DEF(op, op_functor)																			\
		template<class p_scalar_t, typename = enable_if_convertible<p_scalar_t>>											\
		auto operator op (const p_scalar_t& param) const {																		\
			return bi_expr_internal<internal::oper:: op_functor >(internal::scalar_constant<allocator_t>((scalar_t)param));	\
		}

	//----------------------------------------------scalar element-wise operations--------------------------------------------------//
	BC_OPER_SCALAR_BASIC_DEF(+, add)
	BC_OPER_SCALAR_BASIC_DEF(-, sub)
	BC_OPER_SCALAR_BASIC_DEF(*, scalar_mul)
	BC_OPER_SCALAR_BASIC_DEF(>, greater)
	BC_OPER_SCALAR_BASIC_DEF(<, lesser)
	BC_OPER_SCALAR_BASIC_DEF(>=, greater_equal)
	BC_OPER_SCALAR_BASIC_DEF(<=, lesser_equal)
	BC_OPER_SCALAR_BASIC_DEF(==, equal)

	//specialized to upcast  matrix/scalar_value to matrix * (1/scalar_value)
	template<class p_scalar_t, typename = enable_if_convertible<p_scalar_t>>
	auto operator /(const p_scalar_t& param) {
		return bi_expr_internal<internal::oper::scalar_mul>(internal::scalar_constant<allocator_t>((scalar_t)(1/param)));
	}
	//-------------------------------------gemm/gemv/ger-----------------------------------------//
	template<class param_deriv>
	auto operator *(const Tensor_Operations<param_deriv>& param) const {

		static constexpr bool lv_trans  = internal::blas_feature_detector<derived>::transposed;
		static constexpr bool rv_trans  = internal::blas_feature_detector<param_deriv>::transposed;

		static constexpr bool scalmul	= derived::DIMS() == 0 || param_deriv::DIMS() == 0;
		static constexpr bool gemm 		= derived::DIMS() == 2 && param_deriv::DIMS() == 2;
		static constexpr bool gemv 		= derived::DIMS() == 2 && param_deriv::DIMS() == 1;
		static constexpr bool ger  		= derived::DIMS() == 1 && param_deriv::DIMS() == 1 && !lv_trans && rv_trans;
		static constexpr bool dot	  	= derived::DIMS() == 1 && param_deriv::DIMS() == 1 && !ger;
		using matmul_t =
					 std::conditional_t<scalmul, Binary_Expression_t<internal_t_of<param_deriv>, internal::oper::scalar_mul>,
					 std::conditional_t<gemm, 	 Binary_Expression_t<internal_t_of<param_deriv>, internal::oper::gemm<allocator_t>>,
					 std::conditional_t<gemv, 	 Binary_Expression_t<internal_t_of<param_deriv>, internal::oper::gemv<allocator_t>>,
					 std::conditional_t<ger, 	 Binary_Expression_t<internal_t_of<param_deriv>, internal::oper::ger<allocator_t>>,
					 std::conditional_t<dot,	 Binary_Expression_t<internal_t_of<param_deriv>, internal::oper::dot<allocator_t>>, void>>>>>;

		static_assert(!std::is_same<matmul_t, void>::value, "INVALID USE OF OPERATOR *");
		return matmul_t(as_derived().internal(), param.as_derived().internal());
	}

	const auto transpose() const { return un_expr<internal::oper::transpose<allocator_t>>(); }
	 	  auto transpose() 		 { return un_expr<internal::oper::transpose<allocator_t>>(); }

	const auto t() const { return this->transpose(); }
		  auto t()       { return this->transpose(); }


	 //--------------------------------Negation and its Conversions------------------------------//
	 auto operator - () const {
		 return un_expr<internal::oper::negation>();
	 }

private:
	 template<class internal_t>
	 using negated_t = Tensor_Base<internal::Unary_Expression<internal_t, internal::oper::negation>>;
public:
	 //specializations that upcast negation to a 'better' function
	 //(ensures that y -= w * x is as good as y += -(w*x)

	 template<class internal_t>
		derived& operator +=(const negated_t<internal_t>& param) {
			BC_ASSERT_ASSIGNABLE("derived& operator +=(const Tensor_Operations<pDeriv>& param)");
			assert_valid(param);
			evaluate(bi_expr_internal<internal::oper::sub_assign>(param.array));
			return as_derived();
		}
		template<class internal_t>
		derived& operator -=(const negated_t<internal_t>& param) {
			BC_ASSERT_ASSIGNABLE("derived& operator -=(const Tensor_Operations<pDeriv>& param)");
			assert_valid(param);
			evaluate(bi_expr_internal<internal::oper::add_assign>(param.array));
			return as_derived();
		}
		template<class internal_t>
		auto operator +(const negated_t<internal_t>& param) const {
			assert_valid(param);
			return bi_expr_internal<internal::oper::sub>(param.array);
		}


	template<class param_scalar>
	using enable_if_scalar_mul_t = std::enable_if_t<std::is_convertible<param_scalar, scalar_t>::value &&
													!std::is_base_of<BC_Type, param_scalar>::value>;

	//-----------------------------------expression_factory--------------------------------------------------//
	template<class functor>
	auto un_expr(functor f) const {
		return Unary_Expression_t<functor>(as_derived().internal(), f);
	}
	template<class functor>
	const auto un_expr() const {
		return Unary_Expression_t<functor>(as_derived().internal());
	}
	template<class functor, class right_value>
	const auto bi_expr(functor f, const Tensor_Operations<right_value>& rv) const {
		return Binary_Expression_t<internal_t_of<right_value>, functor>(as_derived().internal(), rv.as_derived().internal());
	}
	template<class functor, class right_value>
	const auto bi_expr(const Tensor_Operations<right_value>& rv) const {
		return Binary_Expression_t<internal_t_of<right_value>, functor>(as_derived().internal(), rv.as_derived().internal());
	}
	template<class functor, class right_value>
	const auto bi_expr_internal(functor f, const right_value& rv) const {
		return Binary_Expression_t<right_value, functor>(as_derived().internal(), rv, f);
	}
	template<class functor, class right_value>
	const auto bi_expr_internal(const right_value& rv) const {
		return Binary_Expression_t<right_value, functor>(as_derived().internal(), rv);
	}

	//----------------------------------------------validity checks--------------------------------------------------//
	template<class deriv>  bool non_scalar_op(const Tensor_Operations<deriv>& tensor) const {
		return derived::DIMS() != 0 && deriv::DIMS() != 0;
	}
	template<class deriv>  bool same_rank(const Tensor_Operations<deriv>& tensor) const {
		return derived::DIMS() == deriv::DIMS();
	}
	template<class deriv>  bool same_size(const Tensor_Operations<deriv>& tensor) const {
		return this->as_derived().size() == tensor.as_derived().size();
	}

	//ensures that the smaller tensor is a same-dimensioned "slice" of the other
	template<class deriv>  bool valid_slice(const Tensor_Operations<deriv>& tensor) const {
		constexpr int DIM_MIN = MTF::min(derived::DIMS(), deriv::DIMS());
		for (int i = 0; i < DIM_MIN; ++i)
			if (tensor.as_derived().dimension(i) != as_derived().dimension(i))
				return false;
		return true;
	}

	template<class deriv>
	bool error_message(const Tensor_Operations<deriv>& tensor) const {
		std::cout << "this->DIMS() = " << derived::DIMS() << " this->size() = " <<  as_derived().size() <<  " this_dims ";
		as_derived().print_dimensions();
		std::cout <<  "param->DIMS() = " << deriv::DIMS() << " param.size() = " << tensor.as_derived().size() <<  " param_dims ";
		tensor.as_derived().print_dimensions();
		std::cout << std::endl;
		throw std::invalid_argument("Tensor by Tensor operation - size mismatch - ");
	}

	template<class deriv>
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

	template<class deriv>
	void assert_same_ml(const Tensor_Operations<deriv>& tensor) const {
		static_assert(std::is_same<allocator_t, allocator_t_of<deriv>>::value,
				"TENSOR OPERATIONS BETWEEN THE CPU/GPU ARE PROHIBITED");
	}
public:

	struct Alias{

		derived& tensor;

		Alias(derived& tensor_) : tensor(tensor_) {}

		template<class derived_t>
		void evaluate(const Tensor_Operations<derived_t>& param) {
			internal::Lazy_Evaluator<allocator_t>::evaluate_aliased(static_cast<const derived_t&>(param).internal());
		}

		template<class derived_t>
		auto& operator = (const Tensor_Operations<derived_t>& param) {
			tensor.assert_valid(param);
			evaluate(tensor.bi_expr(internal::oper::assign(), param));
			return tensor;
		}

		template<class derived_t>
		auto& operator += (const Tensor_Operations<derived_t>& param) {
			tensor.assert_valid(param);
			evaluate(tensor.bi_expr(internal::oper::add_assign(), param));
			return tensor.as_derived();
		}

		template<class derived_t>
		auto& operator -= (const Tensor_Operations<derived_t>& param) {
			tensor.assert_valid(param);
			evaluate(tensor.bi_expr(internal::oper::sub_assign(), param));
			return tensor.as_derived();
		}
	};

	friend class Alias;

	Alias alias() {
		return Alias (as_derived());
	}


};


}


template<class p_scalar_t, class p_scalar_t2>
using enable_if_convertible = std::enable_if_t<std::is_convertible<p_scalar_t, p_scalar_t2>::value>;

//----------------------------------------------scalar element-wise operations--------------------------------------------------//
#define BC_OPER_LV_SCALAR_DEF(op, op_functor)																				\
		template<class p_scalar_t, class internal_t, typename = enable_if_convertible<p_scalar_t, typename internal_t::scalar_t>>		\
		 auto operator op (const p_scalar_t& param, const module::Tensor_Operations<internal_t>& tensor) {								\
			using scalar_t = typename internal_t::scalar_t;																				\
			auto scalar_obj = internal::scalar_constant<typename internal_t::allocator_t>((scalar_t)param);								\
			return make_tensor(scalar_obj).bi_expr(internal::oper:: op_functor (), tensor);												\
		}

	BC_OPER_LV_SCALAR_DEF(+, add)
	BC_OPER_LV_SCALAR_DEF(-, sub)
	BC_OPER_LV_SCALAR_DEF(*, scalar_mul)
	BC_OPER_LV_SCALAR_DEF(>, greater)
	BC_OPER_LV_SCALAR_DEF(<, lesser)
	BC_OPER_LV_SCALAR_DEF(>=, greater_equal)
	BC_OPER_LV_SCALAR_DEF(<=, lesser_equal)
	BC_OPER_LV_SCALAR_DEF(==, equal)


	template<class p_scalar_t, class internal_t, typename = enable_if_convertible<p_scalar_t, typename internal_t::scalar_t>>
	 auto operator / (const p_scalar_t& param, const module::Tensor_Operations<internal_t>& tensor) {
		using scalar_t = typename internal_t::scalar_t;
		auto scalar_obj = internal::scalar_constant<typename internal_t::allocator_t>((scalar_t)(1/param));
		return make_tensor(scalar_obj).bi_expr(internal::oper::scalar_mul(), tensor);														\
	}

}

#endif /* TENSOR_CORE_H_ */
