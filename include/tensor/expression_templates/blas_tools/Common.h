/*
 * Common.h
 *
 *  Created on: Apr 24, 2019
 *      Author: joseph
 */

#ifndef BC_BLAS_COMMON_H_
#define BC_BLAS_COMMON_H_

#include "../Array_Scalar_Constant.h"

namespace BC {
namespace exprs {
template<class,class,class> struct Binary_Expression;

namespace blas_tools {

template<class derived>
struct Common_Tools {

	template<class Scalar_t, int alpha_mod, bool lv_scalar, bool rv_scalar, class Context,  class lv_scalar_t, class rv_scalar_t>
	static auto calculate_alpha(Context context, lv_scalar_t lv, rv_scalar_t rv) {

		static constexpr bool lv_host_mode = (BC::exprs::expression_traits<lv_scalar_t>::is_constant);
		static constexpr bool rv_host_mode = (BC::exprs::expression_traits<rv_scalar_t>::is_constant);
		static_assert(lv_host_mode == rv_host_mode || lv_scalar != rv_scalar,
				"Host and Device Scalars may not be mixed Blas calculations");
		static constexpr bool host_mode    = lv_host_mode || rv_host_mode;

		if (host_mode || (!lv_scalar && !rv_scalar)) {
			context.set_blas_pointer_mode_host();
		} else {
			context.set_blas_pointer_mode_device();
		}

		return BC::meta::constexpr_if<!lv_scalar && !rv_scalar>(
					[&](){
						return make_constexpr_scalar<BC::host_tag, alpha_mod, Scalar_t>();
					},
				BC::meta::constexpr_else_if<(lv_scalar != rv_scalar) && (alpha_mod == 1)>(
					[&](){
						return BC::meta::constexpr_ternary<lv_scalar>(
								[&]() { return lv; },
								[&]() { return rv; }
						);
					},
				BC::meta::constexpr_else_if<lv_scalar && rv_scalar>(
					[&]() {
						return BC::meta::constexpr_ternary<host_mode>(
								[&](){
									return make_scalar_constant<BC::host_tag, Scalar_t>(alpha_mod * lv[0] * rv[0]);
								},[&](){
									auto tmp_scalar =  make_temporary_scalar<Scalar_t>(context);
									derived::scalar_multiply(context, tmp_scalar, alpha_mod, lv, rv);
									return tmp_scalar;
								});
					},
				BC::meta::constexpr_else_if<lv_scalar>(
						[&]() {
							return BC::meta::constexpr_if<host_mode>(
									[&](){
										return make_scalar_constant<BC::host_tag, Scalar_t>(alpha_mod * lv[0]);
									},[&](){
										auto tmp_scalar =  make_temporary_scalar<Scalar_t>(context);
										derived::scalar_multiply(context, tmp_scalar, alpha_mod, lv);
										return tmp_scalar;
									});
						}, [&]() { //else if rv_scalar
							return BC::meta::constexpr_if<host_mode>(
									[&](){
										return make_scalar_constant<BC::host_tag, Scalar_t>(alpha_mod * rv[0]);
									},[&](){
										auto tmp_scalar =  make_temporary_scalar<Scalar_t>(context);
										derived::scalar_multiply(context, tmp_scalar, alpha_mod, rv);
										return tmp_scalar;
									});
						})
				)));
	}

	template<int alpha_mod, int beta_mod, class Context, class Lv, class Rv>
	static auto parse_expression(Context context, Lv left, Rv right) {
	    static constexpr bool lv_scalar = blas_expression_traits<Lv>::is_scalar_multiplied;
	    static constexpr bool rv_scalar = blas_expression_traits<Rv>::is_scalar_multiplied;
	    using value_type = typename Lv::value_type;

		auto left_ = greedy_evaluate(blas_expression_traits<Lv>::remove_blas_modifiers(left), context);
	    auto right_ = greedy_evaluate(blas_expression_traits<Rv>::remove_blas_modifiers(right), context);

	    auto alpha_lv = blas_expression_traits<Lv>::get_scalar(left);
		auto alpha_rv = blas_expression_traits<Rv>::get_scalar(right);

		auto alpha_ = calculate_alpha<value_type, alpha_mod, lv_scalar, rv_scalar>(context, alpha_lv, alpha_rv);
	    auto beta_  = make_constexpr_scalar<typename expression_traits<decltype(alpha_)>::allocation_tag, beta_mod, value_type>();//blas_impl::template scalar_constant<value_type, beta_mod>();

	    using left_t = std::remove_reference_t<decltype(left_)>;
	    using right_t = std::remove_reference_t<decltype(right_)>;
	    using alpha_t = std::remove_reference_t<decltype(alpha_)>;
	    using beta_t = std::remove_reference_t<decltype(beta_)>;

		struct contents {
			left_t left;
			right_t right;
			alpha_t alpha;
			beta_t beta;
		};

	    return contents { left_, right_, alpha_, beta_ };
	}
	template<class Context, class Contents>
	static void post_parse_expression_evaluation(Context context, Contents contents) {
		using value_type = typename decltype(contents.alpha)::value_type;
        BC::meta::constexpr_if<(BC::exprs::expression_traits<decltype(contents.alpha)>::is_temporary)>(
            BC::meta::bind([&](auto alpha) {
        		context.template get_allocator_rebound<value_type>().deallocate(alpha, 1);
        	}, 	contents.alpha));

	}
};




}



}}

#endif