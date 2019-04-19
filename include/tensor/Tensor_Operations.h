/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BLACKCAT_TENSOR_OPERATIONS_H_
#define BLACKCAT_TENSOR_OPERATIONS_H_

namespace BC {
template<class expression_t> class Tensor_Base;


namespace module {
template<class derived> class Tensor_Operations;


template<class Expression>
struct Tensor_Operations<Tensor_Base<Expression>> {

    template<class> friend class Tensor_Operations;

    using derived      = Tensor_Base<Expression>;
    using expression_t = Expression;
    using value_type   = typename expression_t::value_type;
    using system_tag   = typename expression_t::system_tag;

private:

    #define BC_ASSERT_ASSIGNABLE(literal) \
    static_assert(exprs::expression_traits<Expression>::is_copy_assignable, \
    		"ASSERT COPY ASSIGNABLE: " literal)

    const derived& as_derived() const { return static_cast<const derived&>(*this); }
          derived& as_derived()       { return static_cast<      derived&>(*this); }

    template<class derived_t>
    void evaluate(const Tensor_Operations<derived_t>& tensor) {
        exprs::evaluate(tensor.as_derived().internal(), this->as_derived().get_context());
    }
public:
    //--------------------------------------assignment operators-----------------------------------------------//
    template<class pDeriv> BCHOT
    derived& operator = (const Tensor_Operations<pDeriv>& param) {
        BC_ASSERT_ASSIGNABLE("derived& operator = (const Tensor_Operations<pDeriv>& param)");
        static_assert(derived::DIMS >= pDeriv::DIMS,
        		"BlackCat_Tensors: Operator= is not a valid operation for (reduction) broadcasting");
        assert_valid(param);
		evaluate(bi_expr< BC::oper::assign >(param));
        return as_derived();
    }

    //specialization for explicit copy operator
    derived& operator = (const BC::meta::only_if<exprs::expression_traits<Expression>::is_copy_assignable, derived>& param) {
        BC_ASSERT_ASSIGNABLE("derived& operator = (const derived& param)");
        assert_valid(param);
        evaluate(bi_expr< oper:: assign >(param));
        return as_derived();
    }


#define BC_OPER_BASIC_ASSIGNMENT_DEF(op, op_functor)                                            	\
                                                                                                	\
    template<class pDeriv> BCHOT                                                                	\
    derived& operator op (const Tensor_Operations<pDeriv>& param) {                             	\
        BC_ASSERT_ASSIGNABLE("derived& operator " #op "(const Tensor_Operations<pDeriv>& param)");  \
        assert_valid(param);                                                                    	\
		using operation = std::conditional_t<(derived::DIMS >= pDeriv::DIMS), 						\
    				oper::op_functor, 																\
    				oper::broadcasted_##op_functor<system_tag>										\
    	>;																							\
		evaluate(bi_expr< operation >(param));                                						\
        return as_derived();                                                                    	\
    }																								\

#define BC_OPER_SCALAR_ASSIGNMENT_DEF(op, op_functor)                                                                \
	template<class p_value_type, class=std::enable_if_t<std::is_convertible<p_value_type, value_type>::value>>       \
	derived& operator  op (const p_value_type& param) {                                                              \
		BC_ASSERT_ASSIGNABLE("derived& operator " #op " (const Tensor_Operations<pDeriv>& param)");                  \
		evaluate(bi_expr_internal<oper:: op_functor >(exprs::make_scalar_constant<system_tag>((value_type)param)));  \
		return as_derived();                                                                                         \
	}

#define BC_OPER_ASSIGNMENT_DEF(op, op_functor)\
    		BC_OPER_SCALAR_ASSIGNMENT_DEF(op, op_functor)\
    		BC_OPER_BASIC_ASSIGNMENT_DEF(op, op_functor)

    BC_OPER_SCALAR_ASSIGNMENT_DEF(=, assign)
    BC_OPER_ASSIGNMENT_DEF(+=, add_assign)
    BC_OPER_ASSIGNMENT_DEF(-=, sub_assign)
    BC_OPER_ASSIGNMENT_DEF(%=, mul_assign)
    BC_OPER_ASSIGNMENT_DEF(/=, div_assign)

#undef BC_OPER_ASSIGNMENT_DEF
#undef BC_OPER_SCALAR_ASSIGNMENT_DEF
#undef BC_OPER_BASIC_ASSIGNMENT_DEF

    //--------------------------------------elementwise expressions-------------------------------//

#define BC_BASIC_COEFFICIENTWISE_DEF(op, op_functor)         \
                                                             \
    template<class pDeriv> BCHOT                             \
    auto op (const Tensor_Operations<pDeriv>& param) const { \
        assert_valid(param);                                 \
        return bi_expr< oper:: op_functor >(param);          \
    }
#define BC_SCALAR_COEFFICIENTWISE_DEF(op, op_functor)                                                                   \
        template<class p_value_type, typename = std::enable_if_t<std::is_convertible<p_value_type, value_type>::value>>                                    \
        auto op (const p_value_type& param) const {                                                                     \
            return bi_expr_internal<oper:: op_functor >(exprs::make_scalar_constant<system_tag>((value_type)param));    \
        }

#define BC_COEFFICIENTWISE_DEF(op, op_functor)\
	BC_BASIC_COEFFICIENTWISE_DEF(op, op_functor)\
    BC_SCALAR_COEFFICIENTWISE_DEF(op, op_functor)

#define BC_OPER_COEFFICIENTWISE_DEF(op, op_functor)\
	BC_BASIC_COEFFICIENTWISE_DEF(operator op, op_functor)\
    BC_SCALAR_COEFFICIENTWISE_DEF(operator op, op_functor)

    BC_OPER_COEFFICIENTWISE_DEF(+, add)
    BC_OPER_COEFFICIENTWISE_DEF(-, sub)
    BC_OPER_COEFFICIENTWISE_DEF(%, mul)
    BC_OPER_COEFFICIENTWISE_DEF(/, div)
    BC_OPER_COEFFICIENTWISE_DEF( == , equal )
    BC_OPER_COEFFICIENTWISE_DEF( >  , greater)
    BC_OPER_COEFFICIENTWISE_DEF( <  , lesser)
    BC_OPER_COEFFICIENTWISE_DEF( >= , greater_equal)
    BC_OPER_COEFFICIENTWISE_DEF( <= , lesser_equal )
    BC_OPER_COEFFICIENTWISE_DEF( && , AND )
    BC_OPER_COEFFICIENTWISE_DEF( || , OR )

    BC_COEFFICIENTWISE_DEF(approx_equal, approx_equal)
    BC_COEFFICIENTWISE_DEF(max_value, max)
    BC_COEFFICIENTWISE_DEF(min_value, min)

#undef BC_BASIC_COEFFICIENTWISE_DEF
#undef BC_SCALAR_COEFFICIENTWISE_DEF
#undef BC_OPER_COEFFICIENTWISE_DEF
#undef BC_COEFFICIENTWISE_DEF

    //----------------------------------------------------------------------------------------------
    //These two functions assist in reordering (or not) a function that is A <Blas_funciton> B * scalar_mul
    //This reorders the scalar mul to be next to the left-value (which allows it to be detected in a BLAS expression-template
private:

    struct reorder_scalar_mul {
    	//lv_t in this instance must be a binary-blas expression
    	template<class matmul_t, class lv_t, class rv_t>
    	static auto impl(lv_t lv, rv_t rv) {
    		auto lv_sub = exprs::make_bin_expr<oper::scalar_mul>(rv, lv.left);
    		auto expr   = exprs::make_bin_expr<typename lv_t::function_t>(lv_sub, lv.right);
    		return make_tensor(expr);
    	}
    };
    struct default_impl {
    	template<class matmul_t, class lv_t, class rv_t>
    	static auto impl(lv_t lv, rv_t rv) {
    		return make_tensor(exprs::make_bin_expr<matmul_t>(lv, rv));
    	}
    };

public:

    //-------------------------------------gemm/gemv/ger-----------------------------------------//
    template<class param_deriv>
    auto operator *(const Tensor_Operations<param_deriv>& param) const {

    	using rv_expression_t = typename Tensor_Operations<param_deriv>::expression_t;
        static constexpr bool lv_trans = exprs::blas_expression_traits<expression_t>::is_transposed;
        static constexpr bool rv_trans = exprs::blas_expression_traits<rv_expression_t>::is_transposed;
        static constexpr bool lv_blas  = oper::operation_traits<Expression>::is_blas_function;

        static constexpr bool scalmul = derived::DIMS == 0 || param_deriv::DIMS == 0;
        static constexpr bool gemm    = derived::DIMS == 2 && param_deriv::DIMS == 2;
        static constexpr bool gemv    = derived::DIMS == 2 && param_deriv::DIMS == 1;
        static constexpr bool ger     = derived::DIMS == 1 && param_deriv::DIMS == 1 && !lv_trans && rv_trans;
        static constexpr bool dot     = derived::DIMS == 1 && param_deriv::DIMS == 1 && !lv_trans && !rv_trans;

        using matmul_t =
                     std::conditional_t<scalmul, oper::scalar_mul,
                     std::conditional_t<gemm,    oper::gemm<system_tag>,
                     std::conditional_t<gemv,    oper::gemv<system_tag>,
                     std::conditional_t<ger,     oper::ger<system_tag>,
                     std::conditional_t<dot,     oper::dot<system_tag>, void>>>>>;

        static_assert(!std::is_void<matmul_t>::value, "INVALID USE OF OPERATOR *");

        using func = std::conditional_t<lv_blas && scalmul, reorder_scalar_mul, default_impl>;
        return func::template impl<matmul_t>(as_derived().internal(), param.as_derived().internal());
    }


    template<class p_value_type,class=std::enable_if_t<std::is_convertible<p_value_type, value_type>::value>>
    auto operator * (const p_value_type& param) const {
        static constexpr bool lv_blas  = oper::operation_traits<expression_t>::is_blas_function;

    	auto scalar_constant = exprs::make_scalar_constant<system_tag>((value_type)param);

        using func = std::conditional_t<lv_blas, reorder_scalar_mul, default_impl>;
        return func::template impl<oper::scalar_mul>(as_derived().internal(), scalar_constant);

    }


    //-------------------------------- Unary Expressions ------------------------------//

    const auto transpose() const { return make_tensor(make_transpose(as_derived().internal())); }
          auto transpose()       { return make_tensor(make_transpose(as_derived().internal())); }

    const auto t() const { return this->transpose(); }
          auto t()       { return this->transpose(); }

	auto operator - () const {
		return un_expr<oper::negation>();
	}


    //-------------------------------- Negation Specializations ------------------------------//

private:
     template<class expression_t>
     using negated_t = Tensor_Base<exprs::Unary_Expression<expression_t, oper::negation>>;
public:

    template<class expression_t>
	derived& operator +=(const negated_t<expression_t>& param) {
		BC_ASSERT_ASSIGNABLE("derived& operator +=(const Tensor_Operations<pDeriv>& param)");
		assert_valid(param);
		evaluate(bi_expr_internal<oper::sub_assign>(param.array));
		return as_derived();
	}

	template<class expression_t>
	derived& operator -=(const negated_t<expression_t>& param) {
		BC_ASSERT_ASSIGNABLE("derived& operator -=(const Tensor_Operations<pDeriv>& param)");
		assert_valid(param);
		evaluate(bi_expr_internal<oper::add_assign>(param.array));
		return as_derived();
	}

	template<class expression_t>
	auto operator +(const negated_t<expression_t>& param) const {
		assert_valid(param);
		return bi_expr_internal<oper::sub>(param.array);
	}

    //-----------------------------------expression_factory--------------------------------------------------//

    template<class functor>
    auto un_expr(functor f) const {
        return make_tensor(exprs::make_un_expr<functor>(as_derived().internal(), f));
    }
    template<class functor>
    const auto un_expr() const {
        return make_tensor(exprs::make_un_expr<functor>(as_derived().internal()));
    }
    template<class functor, class right_value>
    const auto bi_expr(functor f, const Tensor_Operations<right_value>& rv) const {
        return make_tensor(exprs::make_bin_expr<functor>(as_derived().internal(), rv.as_derived().internal()));
    }
    template<class functor, class right_value>
    const auto bi_expr(const Tensor_Operations<right_value>& rv) const {
        return make_tensor(exprs::make_bin_expr<functor>(as_derived().internal(), rv.as_derived().internal()));
    }

private:

    template<class functor, class right_value>
    const auto bi_expr_internal(functor f, const right_value& rv) const {
        return make_tensor(exprs::make_bin_expr<functor>(as_derived().internal(), rv, f));
    }
    template<class functor, class right_value>
    const auto bi_expr_internal(const right_value& rv) const {
        return make_tensor(exprs::make_bin_expr<functor>(as_derived().internal(), rv));
    }

    //----------------------------------------------validity checks--------------------------------------------------//
    template<class deriv>  bool non_scalar_op(const Tensor_Operations<deriv>& tensor) const {
        return derived::DIMS != 0 && deriv::DIMS != 0;
    }
    template<class deriv>  bool same_rank(const Tensor_Operations<deriv>& tensor) const {
        return derived::DIMS == deriv::DIMS;
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
        constexpr BC::size_t  DIM_MIN = meta::min(derived::DIMS, deriv::DIMS);
        for (int i = 0; i < DIM_MIN; ++i)
            if (tensor.as_derived().dimension(i) != as_derived().dimension(i))
                return false;
        return true;
    }

    template<class deriv>
    bool error_message(const Tensor_Operations<deriv>& tensor) const {
        std::cout << "this->DIMS = " << derived::DIMS << " this->size() = " <<  as_derived().size() <<  " this_dims ";
        as_derived().print_dimensions();
        std::cout <<  "param->DIMS = " << deriv::DIMS << " param.size() = " << tensor.as_derived().size() <<  " param_dims ";
        tensor.as_derived().print_dimensions();
        std::cout << std::endl;
        throw std::invalid_argument("Tensor by Tensor operation - size mismatch - ");
    }

    template<class deriv>
    void assert_valid(const Tensor_Operations<deriv>& tensor) const {
#ifdef NDEBUG
    	assert_same_system(tensor);                        //static_assert same allocation (gpu/cpu)
        if (non_scalar_op(tensor)) {                //check if a tensor by scalar operation
            if (same_rank(tensor)) {                //else check is same dimension (element-wise function) (
                if (!same_size(tensor))                    //if is same dimension, ensure same size
                    error_message(tensor);                //else error
                } else if (!valid_slice(tensor)) {    //if not same dimension check if valid slice operation
                    error_message(tensor);            //else error
                }
        }

#endif
    }


public:

    struct Alias{

        derived& tensor;

        Alias(derived& tensor_) : tensor(tensor_) {}

        template<class derived_t>
        void evaluate(const Tensor_Base<derived_t>& param) {
            evaluate_aliased(param.internal(), tensor.get_context());
        }

        template<class derived_t>
        auto& operator = (const Tensor_Operations<derived_t>& param) {
            tensor.assert_valid(param);
            evaluate(tensor.bi_expr(oper::assign(), param));
            return tensor;
        }

        template<class derived_t>
        auto& operator += (const Tensor_Operations<derived_t>& param) {
            tensor.assert_valid(param);
            evaluate(tensor.bi_expr(oper::add_assign(), param));
            return tensor;
        }

        template<class derived_t>
        auto& operator -= (const Tensor_Operations<derived_t>& param) {
            tensor.assert_valid(param);
            evaluate(tensor.bi_expr(oper::sub_assign(), param));
            return tensor;
        }
    };

    friend class Alias;

    Alias alias() {
        return Alias (as_derived());
    }


};
}

//----------------------------------------------scalar element-wise operations--------------------------------------------------//
#define BC_OPER_LV_SCALAR_DEF(op, op_functor)               \
        template<											\
			class p_value_type, 							\
			class expression_t, 							\
        	class = std::enable_if_t<						\
        					std::is_convertible<p_value_type, typename expression_t::value_type>::value && 							 \
        					!BC::exprs::expression_traits<p_value_type>::is_bc_type>							 \
		>        					 																			 \
         auto operator op (const p_value_type& param, const module::Tensor_Operations<expression_t>& tensor) {   \
            using value_type = typename expression_t::value_type;                                                \
            auto scalar_obj = exprs::make_scalar_constant<typename expression_t::system_tag>((value_type)param); \
            return make_tensor(scalar_obj).bi_expr(oper:: op_functor (), tensor);                                \
        }

    BC_OPER_LV_SCALAR_DEF(+, add)
    BC_OPER_LV_SCALAR_DEF(-, sub)
    BC_OPER_LV_SCALAR_DEF(*, scalar_mul)
    BC_OPER_LV_SCALAR_DEF(/, div)
    BC_OPER_LV_SCALAR_DEF(>, greater)
    BC_OPER_LV_SCALAR_DEF(<, lesser)
    BC_OPER_LV_SCALAR_DEF(>=, greater_equal)
    BC_OPER_LV_SCALAR_DEF(<=, lesser_equal)
    BC_OPER_LV_SCALAR_DEF(==, equal)

#undef BC_OPER_LV_SCALAR_DEF

}

#endif /* TENSOR_CORE_H_ */
