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
};

}}}

#endif
