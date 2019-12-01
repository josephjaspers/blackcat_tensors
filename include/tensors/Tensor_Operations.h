/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BLACKCAT_TENSOR_OPERATIONS_H_
#define BLACKCAT_TENSOR_OPERATIONS_H_

#include "expression_templates/Tree_Evaluator.h"

namespace BC {
namespace tensors {

template<class>
class Tensor_Base;

template<class Expression>
struct Tensor_Operations {

	template<class> friend class Tensor_Operations;

	using derived = Tensor_Base<Expression>;
	using expression_t = Expression;
	using value_type = typename expression_t::value_type;
	using system_tag = typename expression_t::system_tag;

private:

	#define BC_ASSERT_ASSIGNABLE(literal) \
	static_assert(exprs::expression_traits<Expression>::is_copy_assignable::value, \
			"ASSERT COPY ASSIGNABLE: " literal)

	const derived& as_derived() const { return static_cast<const derived&>(*this); }
	      derived& as_derived()       { return static_cast<      derived&>(*this); }

	template<class ScalarType>
	using enable_if_scalar = std::enable_if_t<
			std::is_convertible<ScalarType, value_type>::value>;

	template<class derived_t>
	void evaluate(const Tensor_Operations<derived_t>& tensor) {
		BC_ASSERT(this->as_derived().get_stream().get_allocator().allocated_bytes() == 0,
				"Evaluation expects streams allocate_bytes to be 0 pre-evaluation");

		exprs::evaluate(tensor.as_derived().internal(), this->as_derived().get_stream());

		BC_ASSERT(this->as_derived().get_stream().get_allocator().allocated_bytes() == 0,
				"Evaluation expects streams allocate_bytes to be 0 post-evaluation");
	}

public:
	//--------------------------------------assignment operators-----------------------------------------------//
	template<class Xpr> BCHOT
	derived& operator = (const Tensor_Operations<Xpr>& param) {
		BC_ASSERT_ASSIGNABLE("derived& operator = (const Tensor_Operations<Xpr>& param)");
		static_assert(derived::tensor_dimension >= Xpr::tensor_dimension,
				"BlackCat_Tensors: Operator= is not a valid operation for (reduction) broadcasting");
		assert_valid(param);
		evaluate(bi_expr< BC::oper::Assign >(param));
		return as_derived();
	}

	//specialization for explicit copy operator
	derived& operator = (const BC::traits::only_if<exprs::expression_traits<Expression>::is_copy_assignable::value, derived>& param) {
		BC_ASSERT_ASSIGNABLE("derived& operator = (const derived& param)");
		assert_valid(param);
		evaluate(bi_expr< oper::Assign >(param));
		return as_derived();
	}

#define BC_OPER_BASIC_ASSIGNMENT_DEF(op, op_functor)												\
																									\
	template<class Xpr> BCHOT																	\
	derived& operator op (const Tensor_Operations<Xpr>& param) {							 	\
		BC_ASSERT_ASSIGNABLE("derived& operator " #op "(const Tensor_Operations<Xpr>& param)");  \
		assert_valid(param);																		\
		using operation = std::conditional_t<(derived::tensor_dimension >= Xpr::tensor_dimension), 						\
					oper::op_functor##_Assign, 														\
					oper::Atomic_##op_functor<system_tag>>;																							\
		evaluate(bi_expr< operation >(param));														\
		return as_derived();																		\
	}																								\

#define BC_OPER_SCALAR_ASSIGNMENT_DEF(op, op_functor)																\
	template<class ScalarType, class=enable_if_scalar<ScalarType>>	   \
	derived& operator  op (const ScalarType& param) {															  \
		BC_ASSERT_ASSIGNABLE("derived& operator " #op " (const Tensor_Operations<Xpr>& param)");				  \
		evaluate(bi_expr_scalar<oper:: op_functor##_Assign >(exprs::make_scalar_constant<system_tag>((value_type)param)));  \
		return as_derived();																						 \
	}

#define BC_OPER_ASSIGNMENT_DEF(op, op_functor)\
			BC_OPER_SCALAR_ASSIGNMENT_DEF(op, op_functor)\
			BC_OPER_BASIC_ASSIGNMENT_DEF(op, op_functor)


template<class ScalarType, class=enable_if_scalar<ScalarType>>
derived& operator = (const ScalarType& param) {
	BC_ASSERT_ASSIGNABLE("derived& operator =(const Tensor_Operations<Xpr>& param)");
	evaluate(bi_expr_scalar<oper::Assign>(exprs::make_scalar_constant<system_tag>((value_type)param)));
	return as_derived();
}

	BC_OPER_ASSIGNMENT_DEF(+=, Add)
	BC_OPER_ASSIGNMENT_DEF(-=, Sub)
	BC_OPER_ASSIGNMENT_DEF(%=, Mul)
	BC_OPER_ASSIGNMENT_DEF(/=, Div)

#undef BC_OPER_ASSIGNMENT_DEF
#undef BC_OPER_SCALAR_ASSIGNMENT_DEF
#undef BC_OPER_BASIC_ASSIGNMENT_DEF

	//--------------------------------------elementwise expressions-------------------------------//

#define BC_BASIC_COEFFICIENTWISE_DEF(op, op_functor)       \
    template<class Xpr>                                    \
    auto op (const Tensor_Operations<Xpr>& param) const    \
    {                                                      \
        assert_valid(param);                               \
        return bi_expr< oper:: op_functor >(param);        \
    }

#define BC_SCALAR_COEFFICIENTWISE_DEF(op, op_functor)                         \
    template<class ScalarType, class=enable_if_scalar<ScalarType>>            \
    auto op (const ScalarType& param) const                                   \
	{                                                                         \
        return bi_expr_scalar<oper::op_functor>(                              \
                exprs::make_scalar_constant<system_tag>((value_type)param));  \
    }                                                                         \
                                                                              \
    template<class ScalarType, class =enable_if_scalar<ScalarType>>           \
    friend auto op (const ScalarType& param, const Tensor_Operations& tensor) \
	{                                                                         \
		value_type value = param;                                             \
        auto scalar_obj = exprs::make_scalar_constant<system_tag>(value);     \
        return make_tensor(scalar_obj).bi_expr(oper:: op_functor (), tensor); \
    }

#define BC_COEFFICIENTWISE_DEF(op, op_functor)\
    BC_BASIC_COEFFICIENTWISE_DEF(op, op_functor)\
    BC_SCALAR_COEFFICIENTWISE_DEF(op, op_functor)\


#define BC_OPER_COEFFICIENTWISE_DEF(op, op_functor)\
    BC_BASIC_COEFFICIENTWISE_DEF(operator op, op_functor)\
    BC_SCALAR_COEFFICIENTWISE_DEF(operator op, op_functor)\

	BC_OPER_COEFFICIENTWISE_DEF(+, Add)
	BC_OPER_COEFFICIENTWISE_DEF(-, Sub)
	BC_OPER_COEFFICIENTWISE_DEF(%, Mul)
	BC_OPER_COEFFICIENTWISE_DEF(/, Div)
	BC_OPER_COEFFICIENTWISE_DEF( == , Equal )
	BC_OPER_COEFFICIENTWISE_DEF( >  , Greater)
	BC_OPER_COEFFICIENTWISE_DEF( <  , Lesser)
	BC_OPER_COEFFICIENTWISE_DEF( >= , Greater_Equal)
	BC_OPER_COEFFICIENTWISE_DEF( <= , Lesser_Equal )
	BC_OPER_COEFFICIENTWISE_DEF( && , And )
	BC_OPER_COEFFICIENTWISE_DEF( || , Or )

	BC_COEFFICIENTWISE_DEF(approx_equal, Approx_Equal)
	BC_COEFFICIENTWISE_DEF(max_value, Max)
	BC_COEFFICIENTWISE_DEF(min_value, Min)
	BC_SCALAR_COEFFICIENTWISE_DEF(operator *, Scalar_Mul)

#undef BC_BASIC_COEFFICIENTWISE_DEF
#undef BC_SCALAR_COEFFICIENTWISE_DEF
#undef BC_OPER_COEFFICIENTWISE_DEF
#undef BC_COEFFICIENTWISE_DEF

public:

	//-------------------------------------gemm/gemv/ger-----------------------------------------//
	template<class param_deriv>
	auto operator *(const Tensor_Operations<param_deriv>& param) const {

		using rv_expression_t = typename Tensor_Operations<param_deriv>::expression_t;
		static constexpr bool lv_trans = exprs::blas_expression_traits<expression_t>::is_transposed::value;
		static constexpr bool rv_trans = exprs::blas_expression_traits<rv_expression_t>::is_transposed::value;

		static constexpr bool scalmul = derived::tensor_dimension == 0 || param_deriv::tensor_dimension == 0;
		static constexpr bool gemm	= derived::tensor_dimension == 2 && param_deriv::tensor_dimension == 2;
		static constexpr bool gemv	= derived::tensor_dimension == 2 && param_deriv::tensor_dimension == 1;
		static constexpr bool ger	 = derived::tensor_dimension == 1 && param_deriv::tensor_dimension == 1 && !lv_trans && rv_trans;
		static constexpr bool dot	 = derived::tensor_dimension == 1 && param_deriv::tensor_dimension == 1 && !lv_trans && !rv_trans;

		using matmul_t =
					 std::conditional_t<scalmul, oper::Scalar_Mul,
					 std::conditional_t<gemm,	oper::gemm<system_tag>,
					 std::conditional_t<gemv,	oper::gemv<system_tag>,
					 std::conditional_t<ger,	 oper::ger<system_tag>,
					 std::conditional_t<dot,	 oper::dot<system_tag>, void>>>>>;

		static_assert(!std::is_void<matmul_t>::value, "INVALID USE OF OPERATOR *");
		return bi_expr<matmul_t>(param);
	}

	//-------------------------------- Unary Expressions ------------------------------//

	const auto transpose() const { return make_tensor(make_transpose(as_derived().internal())); }
		  auto transpose()	   { return make_tensor(make_transpose(as_derived().internal())); }

	const auto t() const { return this->transpose(); }
		  auto t()	   { return this->transpose(); }

	auto operator - () const {
		return un_expr<oper::negation>();
	}


	//-------------------------------- Negation Specializations ------------------------------//

private:
	 template<class Xpr>
	 using negated_t = Tensor_Base<exprs::Unary_Expression<oper::negation, Xpr>>;
public:

	template<class Xpr>
	derived& operator +=(const negated_t<Xpr>& param) {
		BC_ASSERT_ASSIGNABLE("derived& operator +=(const Tensor_Operations<Xpr>& param)");
		assert_valid(param);
		evaluate(bi_expr<oper::Sub_Assign>(make_tensor(param.array)));
		return as_derived();
	}

	template<class Xpr>
	derived& operator -=(const negated_t<Xpr>& param) {
		BC_ASSERT_ASSIGNABLE("derived& operator -=(const Tensor_Operations<Xpr>& param)");
		assert_valid(param);
		evaluate(bi_expr<oper::Add_Assign>(make_tensor(param.array)));
		return as_derived();
	}

	template<class Xpr>
	auto operator +(const negated_t<Xpr>& param) const {
		assert_valid(param);
		return bi_expr<oper::Sub>(make_tensor(param.array));
	}

	//-----------------------------------expression_factory--------------------------------------------------//

	template<class functor>
	auto un_expr(functor f) const {
		return make_tensor(exprs::make_un_expr<functor>(as_derived().internal(), f));
	}
	template<class functor>
	auto un_expr() const {
		return make_tensor(exprs::make_un_expr<functor>(as_derived().internal()));
	}
	template<class functor, class Xpr>
	auto bi_expr(functor f, const Tensor_Operations<Xpr>& rv) const {
		return make_tensor(exprs::make_bin_expr<functor>(as_derived().internal(), rv.as_derived().internal()));
	}
	template<class functor, class Xpr>
	auto bi_expr(const Tensor_Operations<Xpr>& rv) const {
		return make_tensor(exprs::make_bin_expr<functor>(as_derived().internal(), rv.as_derived().internal()));
	}
	template<class functor, class Xpr>
	auto bi_expr(functor f, const Tensor_Operations<Xpr>& rv) {
		return make_tensor(exprs::make_bin_expr<functor>(as_derived().internal(), rv.as_derived().internal()));
	}
	template<class functor, class Xpr>
	auto bi_expr(const Tensor_Operations<Xpr>& rv) {
		return make_tensor(exprs::make_bin_expr<functor>(as_derived().internal(), rv.as_derived().internal()));
	}

private:

	template<class functor, class Scalar>
	auto bi_expr_scalar(functor f, const Scalar& rv) const {
		return make_tensor(exprs::make_bin_expr<functor>(as_derived().internal(), rv, f));
	}
	template<class functor, class Scalar>
	auto bi_expr_scalar(const Scalar& rv) const {
		return make_tensor(exprs::make_bin_expr<functor>(as_derived().internal(), rv));
	}

	//ensures that the smaller tensor is a same-dimensioned "slice" of the other
	template<class Xpr>
	bool valid_slice(const Tensor_Operations<Xpr>& tensor) const {
		constexpr BC::size_t min_dim =
				traits::min(derived::tensor_dimension, Xpr::tensor_dimension);

		for (int i = 0; i < min_dim; ++i)
			if (tensor.as_derived().dimension(i) != as_derived().dimension(i))
				return false;
		return true;
	}

	template<class deriv>
	bool error_message(const Tensor_Operations<deriv>& tensor) const {
		std::cout << "this->tensor_dimension = " << derived::tensor_dimension << " this->size() = " <<  as_derived().size() <<  " this_dims ";
		as_derived().print_dimensions();
		std::cout <<  "param->tensor_dimension = " << deriv::tensor_dimension << " param.size() = " << tensor.as_derived().size() <<  " param_dims ";
		tensor.as_derived().print_dimensions();
		std::cout << std::endl;
		throw std::invalid_argument("Tensor by Tensor operation - size mismatch - ");
	}

	template<class Xpr>
	void assert_valid(const Tensor_Operations<Xpr>& tensor) const {
		static_assert(std::is_same<system_tag, typename Xpr::system_tag>::value,
				"Operations between two tensors must have the same system_tag");

		bool scalar_op =
				expression_t::tensor_dimension == 0 ||  Xpr::tensor_dimension == 0;
		bool same_dimension =
				expression_t::tensor_dimension == Xpr::tensor_dimension;
		bool same_size = as_derived().size() == tensor.as_derived().size();


		if (!scalar_op) {				//check if a tensor by scalar operation
			if (same_dimension) {				//else check is same dimension (element-wise function) (
				if (!same_size)			 //if is same dimension, ensure same size
					error_message(tensor);		  //else error
			} else if (!valid_slice(tensor)) {  //if not same dimension check if valid slice operation
				error_message(tensor);		  //else error
			}
		}
	}


public:

	struct Alias{

		derived& tensor;

		Alias(derived& tensor_) : tensor(tensor_) {}

		template<class derived_t>
		void evaluate(const Tensor_Base<derived_t>& param) {
			evaluate_aliased(param.internal(), tensor.get_stream());
		}

		template<class derived_t>
		auto& operator = (const Tensor_Operations<derived_t>& param) {
			tensor.assert_valid(param);
			evaluate(tensor.bi_expr(oper::Assign(), param));
			return tensor;
		}

		template<class derived_t>
		auto& operator += (const Tensor_Operations<derived_t>& param) {
			tensor.assert_valid(param);
			evaluate(tensor.bi_expr(oper::Add_Assign(), param));
			return tensor;
		}

		template<class derived_t>
		auto& operator -= (const Tensor_Operations<derived_t>& param) {
			tensor.assert_valid(param);
			evaluate(tensor.bi_expr(oper::Sub_Assign(), param));
			return tensor;
		}
	};

	friend class Alias;

	Alias alias() {
		return Alias (as_derived());
	}


};

//----------------------------------------------scalar element-wise operations--------------------------------------------------//

#undef BC_ASSERT_ASSIGNABLE

	template<class Expression>
	auto sum(const Tensor_Base<Expression>& tensor) {
		return tensor.template un_expr<exprs::Sum<typename Expression::system_tag>>();
	}
}
}

#endif /* TENSOR_CORE_H_ */
