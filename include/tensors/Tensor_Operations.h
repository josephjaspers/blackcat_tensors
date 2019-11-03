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

	template<class derived_t>
	void evaluate(Tensor_Operations<derived_t>&& tensor) {
		BC_ASSERT(this->as_derived().get_stream().get_allocator().allocated_bytes() == 0,
				"Evaluation expects streams allocate_bytes to be 0 pre-evaluation");
		exprs::evaluate(tensor.as_derived().internal(), this->as_derived().get_stream());

		BC_ASSERT(this->as_derived().get_stream().get_allocator().allocated_bytes() == 0,
				"Evaluation expects streams allocate_bytes to be 0 post-evaluation");
	}

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
	template<class pDeriv> BCHOT
	derived& operator = (const Tensor_Operations<pDeriv>& param) {
		BC_ASSERT_ASSIGNABLE("derived& operator = (const Tensor_Operations<pDeriv>& param)");
		static_assert(derived::tensor_dimension >= pDeriv::tensor_dimension,
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
	template<class pDeriv> BCHOT
	derived& assign(const Tensor_Operations<pDeriv>& param) {
		BC_ASSERT_ASSIGNABLE("derived& operator = (const Tensor_Operations<pDeriv>& param)");
		static_assert(derived::tensor_dimension >= pDeriv::tensor_dimension,
				"BlackCat_Tensors: Operator= is not a valid operation for (reduction) broadcasting");
		assert_valid(param);
		evaluate(bi_expr< BC::oper::Assign >(param));
		return as_derived();
	}

	//specialization for explicit copy operator
	derived& assign(const BC::traits::only_if<exprs::expression_traits<Expression>::is_copy_assignable::value, derived>& param) {
		BC_ASSERT_ASSIGNABLE("derived& operator = (const derived& param)");
		assert_valid(param);
		evaluate(bi_expr< oper::Assign >(param));
		return as_derived();
	}

#define BC_OPER_BASIC_ASSIGNMENT_DEF(op, op_functor)												\
																									\
	template<class pDeriv> BCHOT																	\
	derived& operator op (const Tensor_Operations<pDeriv>& param) {							 	\
		BC_ASSERT_ASSIGNABLE("derived& operator " #op "(const Tensor_Operations<pDeriv>& param)");  \
		assert_valid(param);																		\
		using operation = std::conditional_t<(derived::tensor_dimension >= pDeriv::tensor_dimension), 						\
					oper::op_functor##_Assign, 														\
					oper::Atomic_##op_functor<system_tag>														\
		>;																							\
		evaluate(bi_expr< operation >(param));														\
		return as_derived();																		\
	}																								\

#define BC_OPER_SCALAR_ASSIGNMENT_DEF(op, op_functor)																\
	template<class p_value_type, class=std::enable_if_t<std::is_convertible<p_value_type, value_type>::value>>	   \
	derived& operator  op (const p_value_type& param) {															  \
		BC_ASSERT_ASSIGNABLE("derived& operator " #op " (const Tensor_Operations<pDeriv>& param)");				  \
		evaluate(bi_expr_scalar<oper:: op_functor##_Assign >(exprs::make_scalar_constant<system_tag>((value_type)param)));  \
		return as_derived();																						 \
	}

#define BC_NAMED_ASSIGNMENT_OPER(op, op_functor)\
		template<class Arg>\
		derived& op_functor##_assign (const Arg& arg) {\
			return *this op arg;\
		}


#define BC_OPER_ASSIGNMENT_DEF(op, op_functor)\
			BC_OPER_SCALAR_ASSIGNMENT_DEF(op, op_functor)\
			BC_OPER_BASIC_ASSIGNMENT_DEF(op, op_functor)


template<class p_value_type, class=std::enable_if_t<std::is_convertible<p_value_type, value_type>::value>>
derived& operator = (const p_value_type& param) {
	BC_ASSERT_ASSIGNABLE("derived& operator =(const Tensor_Operations<pDeriv>& param)");
	evaluate(bi_expr_scalar<oper::Assign>(exprs::make_scalar_constant<system_tag>((value_type)param)));
	return as_derived();
}

template<class p_value_type, class=std::enable_if_t<std::is_convertible<p_value_type, value_type>::value>>
derived& assign(const p_value_type& param) {
	BC_ASSERT_ASSIGNABLE("derived& operator =(const Tensor_Operations<pDeriv>& param)");
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

#define BC_BASIC_COEFFICIENTWISE_DEF(op, op_functor)		 \
															 \
	template<class pDeriv> BCHOT							 \
	auto op (const Tensor_Operations<pDeriv>& param) const { \
		assert_valid(param);								 \
		return bi_expr< oper:: op_functor >(param);		  \
	}
#define BC_SCALAR_COEFFICIENTWISE_DEF(op, op_functor)																   \
		template<class p_value_type, typename = std::enable_if_t<std::is_convertible<p_value_type, value_type>::value>>									\
		auto op (const p_value_type& param) const {																	 \
			return bi_expr_scalar<oper:: op_functor >(exprs::make_scalar_constant<system_tag>((value_type)param));	\
		}

#define BC_COEFFICIENTWISE_DEF(op, op_functor)\
	BC_BASIC_COEFFICIENTWISE_DEF(op, op_functor)\
	BC_SCALAR_COEFFICIENTWISE_DEF(op, op_functor)

#define BC_OPER_COEFFICIENTWISE_DEF(op, op_functor)\
	BC_BASIC_COEFFICIENTWISE_DEF(operator op, op_functor)\
	BC_SCALAR_COEFFICIENTWISE_DEF(operator op, op_functor)

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
	 template<class expression_t>
	 using negated_t = Tensor_Base<exprs::Unary_Expression<oper::negation, expression_t>>;
public:

	template<class expression_t>
	derived& operator +=(const negated_t<expression_t>& param) {
		BC_ASSERT_ASSIGNABLE("derived& operator +=(const Tensor_Operations<pDeriv>& param)");
		assert_valid(param);
		evaluate(bi_expr<oper::sub_assign>(param.array));
		return as_derived();
	}

	template<class expression_t>
	derived& operator -=(const negated_t<expression_t>& param) {
		BC_ASSERT_ASSIGNABLE("derived& operator -=(const Tensor_Operations<pDeriv>& param)");
		assert_valid(param);
		evaluate(bi_expr<oper::add_assign>(param.array));
		return as_derived();
	}

	template<class expression_t>
	auto operator +(const negated_t<expression_t>& param) const {
		assert_valid(param);
		return bi_expr<oper::sub>(param.array);
	}

	// --------------- host_to_device/device_to_host copy function --------------- //

	template<class rightDeriv>
	auto multichannel_conv2d(const Tensor_Operations<rightDeriv>& rv) const {
		return bi_expr<BC::tensors::exprs::multichannel_conv2d> (rv);
	}

	template<class rightDeriv>
	auto multichannel_conv2d_data_backwards(const Tensor_Operations<rightDeriv>& rv) const {
		return bi_expr<BC::tensors::exprs::multichannel_conv2d_data_backwards> (rv);
	}

	template<class rightDeriv>
	auto multichannel_conv2d_kernel_backwards(const Tensor_Operations<rightDeriv>& rv) const {
		return bi_expr<BC::tensors::exprs::multichannel_conv2d_kernel_backwards> (rv);
	}

	auto img2col() const {
		return un_expr<BC::tensors::exprs::img2col>();
	}

	template<class right_value>
	void copy(const Tensor_Operations<right_value>& rv) {
		static_assert(exprs::expression_traits<Expression>::is_copy_assignable::value, "copy lv must be array");
		static_assert(exprs::expression_traits<right_value>::is_copy_assignable::value, "copy rv most be array");
		static_assert(Expression::tensor_iterator_dimension <= 1, "copy only accepts continuous");
		static_assert(right_value::tensor_iterator_dimension <= 1, "copy only accepts continuous");

		if (!same_size(rv)) {
			std::cout << "Attempting to copy two different size tensors (ERROR)"  << std::endl;
			throw 1;
		}

#ifdef __CUDACC__
		using copy_impl = BC::utility::implementation<device_tag>;

		if (std::is_same<system_tag, typename right_value::system_tag>::value) {
			//Ensures it only compiles when true
			BC::traits::constexpr_if<std::is_same<system_tag, typename right_value::system_tag>::value>(
					BC::traits::bind([](auto& self, const auto& rv){
						self = rv;
			}, *this, rv));
		} else if (std::is_same<system_tag, device_tag>::value) {
			copy_impl::HostToDevice(as_derived().data(),
					rv.as_derived().data(),
					as_derived().size());
		} else {
			copy_impl::DeviceToHost(as_derived().data(),
					rv.as_derived().data(),
					as_derived().size());
		}
#else
		this->as_derived() = rv.as_derived();
#endif
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
	template<class functor, class right_value>
	auto bi_expr(functor f, const Tensor_Operations<right_value>& rv) const {
		return make_tensor(exprs::make_bin_expr<functor>(as_derived().internal(), rv.as_derived().internal()));
	}
	template<class functor, class right_value>
	auto bi_expr(const Tensor_Operations<right_value>& rv) const {
		return make_tensor(exprs::make_bin_expr<functor>(as_derived().internal(), rv.as_derived().internal()));
	}
	template<class functor, class right_value>
	auto bi_expr(functor f, const Tensor_Operations<right_value>& rv) {
		return make_tensor(exprs::make_bin_expr<functor>(as_derived().internal(), rv.as_derived().internal()));
	}
	template<class functor, class right_value>
	auto bi_expr(const Tensor_Operations<right_value>& rv) {
		return make_tensor(exprs::make_bin_expr<functor>(as_derived().internal(), rv.as_derived().internal()));
	}

private:

	template<class functor, class right_value>
	auto bi_expr_scalar(functor f, const right_value& rv) const {
		return make_tensor(exprs::make_bin_expr<functor>(as_derived().internal(), rv, f));
	}
	template<class functor, class right_value>
	auto bi_expr_scalar(const right_value& rv) const {
		return make_tensor(exprs::make_bin_expr<functor>(as_derived().internal(), rv));
	}

	//----------------------------------------------validity checks--------------------------------------------------//
	template<class deriv>  bool non_scalar_op(const Tensor_Operations<deriv>& tensor) const {
		return derived::tensor_dimension != 0 && deriv::tensor_dimension != 0;
	}
	template<class deriv>  bool same_rank(const Tensor_Operations<deriv>& tensor) const {
		return derived::tensor_dimension == deriv::tensor_dimension;
	}
	template<class deriv>  bool same_size(const Tensor_Operations<deriv>& tensor) const {
		return this->as_derived().size() == tensor.as_derived().size();
	}
	template<class deriv>
	void assert_same_system(const Tensor_Operations<deriv>& tensor) const {
		static_assert(std::is_same<typename deriv::system_tag, system_tag>::value,
				"TENSOR OPERATIONS BETWEEN THE CPU/GPU ARE PROHIBITED");
	}

	//ensures that the smaller tensor is a same-dimensioned "slice" of the other
	template<class deriv>  bool valid_slice(const Tensor_Operations<deriv>& tensor) const {
		constexpr BC::size_t  DIM_MIN = traits::min(derived::tensor_dimension, deriv::tensor_dimension);
		for (int i = 0; i < DIM_MIN; ++i)
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

	template<class deriv>
	void assert_valid(const Tensor_Operations<deriv>& tensor) const {
		assert_same_system(tensor); //static_assert same allocation (gpu/cpu)
		if (non_scalar_op(tensor)) {				//check if a tensor by scalar operation
			if (same_rank(tensor)) {				//else check is same dimension (element-wise function) (
				if (!same_size(tensor))			 //if is same dimension, ensure same size
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
#define BC_OPER_LV_SCALAR_DEF(op, op_functor)			   \
		template<											\
			class p_value_type, 							\
			class expression_t, 							\
			class = std::enable_if_t<						\
							std::is_convertible<p_value_type, typename expression_t::value_type>::value && 							 \
							!BC::tensors::exprs::expression_traits<p_value_type>::is_expression_template::value>							 \
		>							 																			 \
		 auto operator op (const p_value_type& param, const Tensor_Base<expression_t>& tensor) {   \
			using value_type = typename expression_t::value_type;												\
			auto scalar_obj = exprs::make_scalar_constant<typename expression_t::system_tag>((value_type)param); \
			return make_tensor(scalar_obj).bi_expr(oper:: op_functor (), tensor);								\
		}

	BC_OPER_LV_SCALAR_DEF(+, Add)
	BC_OPER_LV_SCALAR_DEF(-, Sub)
	BC_OPER_LV_SCALAR_DEF(*, Scalar_Mul)
	BC_OPER_LV_SCALAR_DEF(/, Div)
	BC_OPER_LV_SCALAR_DEF(>, Greater)
	BC_OPER_LV_SCALAR_DEF(<, Lesser)
	BC_OPER_LV_SCALAR_DEF(>=, Greater_Equal)
	BC_OPER_LV_SCALAR_DEF(<=, Lesser_Equal)
	BC_OPER_LV_SCALAR_DEF(==, Equal)

#undef BC_OPER_LV_SCALAR_DEF
#undef BC_ASSERT_ASSIGNABLE

	template<class Expression>
	auto sum(const Tensor_Base<Expression>& tensor) {
		return tensor.template un_expr<exprs::Sum<typename Expression::system_tag>>();
	}
}
}

#endif /* TENSOR_CORE_H_ */
