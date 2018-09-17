/*
 * Array.h
 *
 *  Created on: Dec 30, 2017
 *      Author: joseph
 */

#ifndef TENSOR_HEAD_H_
#define TENSOR_HEAD_H_

#include "Tensor_Common.h"
#include "Expression_Templates/Expression_Unary.h"
#include "Expression_Templates/Expression_Binary.h"
#include "Expression_Templates/Operations/Binary.h"
#include "Expression_Templates/Operations/Unary.h"
#include "Expression_Templates/Operations/BLAS.h"

#include "Expression_Templates/Function_gemm.h"
#include "Expression_Templates/Function_gemv.h"
#include "Expression_Templates/Function_ger.h"
#include "Expression_Templates/Function_dot.h"
#include "Expression_Templates/Function_transpose.h"

#include "Expression_Templates/Tree_Evaluator_Runner.h"


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
	using mathlib_t 		= typename internal_t::mathlib_t;

	template<class deriv>
	using internal_t_of		= typename Tensor_Operations<deriv>::internal_t;

	template<class deriv>
	using mathlib_t_of		= typename Tensor_Operations<deriv>::mathlib_t;

	static constexpr bool	copy_assignable = BC_array_copy_assignable<internal_type>();
	#define BC_ASSERT_ASSIGNABLE(literal) static_assert(copy_assignable, "ASSERT COPY ASSIGNABLE: " literal)

	template<class expr> 		   using unary_expression_t  = BC::Tensor_Base<internal::unary_expression<internal_t, expr>>;
	template<class rv, class expr> using binary_expression_t = BC::Tensor_Base<internal::binary_expression<internal_t, rv, expr>>;

	const derived& as_derived() const { return static_cast<const derived&>(*this); }
	 	  derived& as_derived() 	  { return static_cast<	     derived&>(*this); }

	//--------------------------------------evaluation implementation-----------------------------------------------//
	template<class derived_t>
	void evaluate(const Tensor_Operations<derived_t>& tensor) {
		internal::evaluate(tensor.as_derived().internal());
	}

public:

	//--------------------------------------assignment operators-----------------------------------------------//
	template<class pDeriv>
	derived& operator =(const Tensor_Operations<pDeriv>& param) {
		BC_ASSERT_ASSIGNABLE("derived& operator =(const Tensor_Operations<pDeriv>& param)");
		assert_valid(param);
		evaluate(bi_expr<internal::oper::assign>(param));
		return as_derived();
	}
	template<class pDeriv>
	derived& operator +=(const Tensor_Operations<pDeriv>& param) {
		BC_ASSERT_ASSIGNABLE("derived& operator +=(const Tensor_Operations<pDeriv>& param)");
		assert_valid(param);
		evaluate(bi_expr<internal::oper::add_assign>(param));
		return as_derived();
	}
	template<class pDeriv>
	derived& operator -=(const Tensor_Operations<pDeriv>& param) {
		BC_ASSERT_ASSIGNABLE("derived& operator -=(const Tensor_Operations<pDeriv>& param)");
		assert_valid(param);
		evaluate(bi_expr<internal::oper::sub_assign>(param));
		return as_derived();
	}
	template<class pDeriv>
	derived& operator /=(const Tensor_Operations<pDeriv>& param) {
		BC_ASSERT_ASSIGNABLE("derived& operator /=(const Tensor_Operations<pDeriv>& param)");
		assert_valid(param);
		evaluate(bi_expr<internal::oper::div_assign>(param));
		return as_derived();
	}
	//pointwise multiply
	template<class pDeriv>
	derived& operator %=(const Tensor_Operations<pDeriv>& param) {
		BC_ASSERT_ASSIGNABLE("derived& operator %=(const Tensor_Operations<pDeriv>& param)");
		assert_valid(param);
		evaluate(bi_expr<internal::oper::mul_assign>(param));
		return as_derived();
	}
	//-------------------------------------gemm/gemv/ger-----------------------------------------//
	template<class param_deriv>
	auto operator *(const Tensor_Operations<param_deriv>& param) const {

		static constexpr bool scalmul	= derived::DIMS() == 0 || param_deriv::DIMS() == 0;
		static constexpr bool gemm 		= derived::DIMS() == 2 && param_deriv::DIMS() == 2;
		static constexpr bool gemv 		= derived::DIMS() == 2 && param_deriv::DIMS() == 1;
		static constexpr bool ger  		= derived::DIMS() == 1 && param_deriv::DIMS() == 1 && internal::blas_feature_detector<param_deriv>::transposed;
		static constexpr bool dot		= derived::DIMS() == 1 && param_deriv::DIMS() == 1 && !ger;
		using matmul_t =
					 std::conditional_t<scalmul, binary_expression_t<internal_t_of<param_deriv>, internal::oper::scalar_mul>,
					 std::conditional_t<gemm, 	 binary_expression_t<internal_t_of<param_deriv>, internal::oper::gemm<mathlib_t>>,
					 std::conditional_t<gemv, 	 binary_expression_t<internal_t_of<param_deriv>, internal::oper::gemv<mathlib_t>>,
					 std::conditional_t<ger, 	 binary_expression_t<internal_t_of<param_deriv>, internal::oper::ger<mathlib_t>>,
					 std::conditional_t<dot,	 binary_expression_t<internal_t_of<param_deriv>, internal::oper::dot<mathlib_t>>, void>>>>>;

		static_assert(!std::is_same<matmul_t, void>::value, "INVALID USE OF OPERATOR *");

		return matmul_t(as_derived().internal(), param.as_derived().internal());
	}

	//--------------------------------------pointwise operators-------------------------------//
	template<class pDeriv>
	auto operator +(const Tensor_Operations<pDeriv>& param) const {
		assert_valid(param);
		return bi_expr<internal::oper::add>(param);
	}
	template<class pDeriv>
	auto operator -(const Tensor_Operations<pDeriv>& param) const {
		assert_valid(param);
		return bi_expr<internal::oper::sub>(param);
	}
	template<class pDeriv>
	auto operator /(const Tensor_Operations<pDeriv>& param) const {
		assert_valid(param);
		return bi_expr<internal::oper::div>(param);
	}
	//pointwise multiply
	template<class pDeriv>
	auto operator %(const Tensor_Operations<pDeriv>& param) const {
		assert_valid(param);
		return bi_expr<internal::oper::mul>(param);
	}


	 //--------------------------------Other Operators------------------------------//

	 auto operator - () const {
		 return un_expr<internal::oper::negation>();
	 }
	const auto transpose() const { return un_expr<internal::oper::transpose<mathlib_t>>(); }
	 	  auto transpose() 		 { return un_expr<internal::oper::transpose<mathlib_t>>(); }

	const auto t() const { return transpose(); }
		  auto t()       { return transpose(); }

	template<class pDeriv>
	auto operator ==(const Tensor_Operations<pDeriv>& param) const {
		assert_valid(param);
		return bi_expr<internal::oper::equal>(param);
	}
	template<class pDeriv>
	auto operator >(const Tensor_Operations<pDeriv>& param) const {
		assert_valid(param);
		return bi_expr<internal::oper::greater>(param);
	}
	template<class pDeriv>
	auto operator <(const Tensor_Operations<pDeriv>& param) const {
		assert_valid(param);
		return bi_expr<internal::oper::lesser>(param);
	}
	template<class pDeriv>
	auto operator >=(const Tensor_Operations<pDeriv>& param) const {
		assert_valid(param);
		return bi_expr<internal::oper::greater_equal>(param);
	}
	template<class pDeriv>
	auto operator <=(const Tensor_Operations<pDeriv>& param) const {
		assert_valid(param);
		return bi_expr<internal::oper::lesser_equal>(param);
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
		return binary_expression_t<internal_t_of<right_value>, functor>(as_derived().internal(), rv.as_derived().internal());
	}
	template<class functor, class right_value>
	const auto bi_expr(const Tensor_Operations<right_value>& rv) const {
		return binary_expression_t<internal_t_of<right_value>, functor>(as_derived().internal(), rv.as_derived().internal());
	}
	 //--------------------------------ASSERTIONS------------------------------//


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
		static_assert(std::is_same<mathlib_t_of<derived>, mathlib_t_of<deriv>>::value,
				"TENSOR OPERATIONS BETWEEN THE CPU/GPU ARE PROHIBITED");
	}
public:
	auto _normalize(scalar_t min, scalar_t max) const {
		return un_expr(internal::oper::norm<scalar_t>(scalar_t(min), scalar_t(max)));
	}
	static auto fix(Tensor_Operations& tensor) {
		return tensor.un_expr(internal::oper::fix());
	}
	static auto abs(const Tensor_Operations& tensor) {
		return tensor.un_expr(internal::oper::absolute());
	}


	struct Alias{

		derived& tensor;

		Alias(derived& tensor_) : tensor(tensor_) {}

		template<class derived_t>
		void evaluate(const Tensor_Operations<derived_t>& param) {
			internal::Lazy_Evaluator<mathlib_t>::evaluate_aliased(static_cast<const derived_t&>(param).internal());
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

template<class internal_t, class min_, class max_>
static auto normalize(const Tensor_Operations<internal_t>& tensor, min_ min, max_ max) {
	return tensor._normalize(min, max);
}

}



}

#endif /* TENSOR_CORE_H_ */
