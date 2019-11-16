/*
 * Common.h
 *
 *  Created on: Apr 24, 2019
 *	  Author: joseph
 */

#ifndef BC_BLAS_COMMON_H_
#define BC_BLAS_COMMON_H_

#include "../Array_Scalar_Constant.h"

namespace BC {
namespace tensors {
namespace exprs { 

template<class,class,class>
struct Binary_Expression;

namespace blas_expression_parser {

template<class derived>
struct Common_Tools {


private:

	template<class ValueType, class Stream>
	static auto make_kernel_scalar(Stream stream) {
		using system_tag = typename Stream::system_tag;
		using Array = Kernel_Array<Shape<0>, ValueType, system_tag, temporary_tag>;
		return Array(BC::Shape<0>(), stream.template get_allocator_rebound<ValueType>().allocate(1));
	}

public:

	template<class ValueType, int alpha_mod, bool lv_scalar, bool rv_scalar, class Stream, class lv_scalar_t, class rv_scalar_t>
	static auto calculate_alpha(Stream stream, lv_scalar_t lv, rv_scalar_t rv) {

		static constexpr bool lv_host_mode = BC::tensors::exprs::expression_traits<lv_scalar_t>::is_stack_allocated::value;
		static constexpr bool rv_host_mode = BC::tensors::exprs::expression_traits<rv_scalar_t>::is_stack_allocated::value;

		static_assert(lv_host_mode == rv_host_mode || lv_scalar != rv_scalar,
				"Host and Device Scalars may not be mixed Blas calculations");
		static constexpr bool host_mode	= lv_host_mode || rv_host_mode;

		if (host_mode || (!lv_scalar && !rv_scalar)) {
			stream.set_blas_pointer_mode_host();
		} else {
			stream.set_blas_pointer_mode_device();
		}

		return BC::traits::constexpr_if<(lv_scalar != rv_scalar) && (alpha_mod == 1)>(
					[&](){
						return BC::traits::constexpr_ternary<lv_scalar>(
								[=]() { return lv; },
								[=]() { return rv; }
						);
				},
				BC::traits::constexpr_else_if<!lv_scalar && !rv_scalar>(
				[&](){
					return make_constexpr_scalar<BC::host_tag, alpha_mod, ValueType>();
				},
				BC::traits::constexpr_else_if<lv_scalar && rv_scalar>(
					[&]() {
						return BC::traits::constexpr_ternary<host_mode>(
								[&](){
									return make_scalar_constant<BC::host_tag, ValueType>(alpha_mod * lv[0] * rv[0]);
								},[&](){
									auto tmp_scalar = make_kernel_scalar<ValueType>(stream);
									derived::scalar_multiply(stream, tmp_scalar, alpha_mod, lv, rv);
									return tmp_scalar;
								});
					},
				BC::traits::constexpr_else_if<lv_scalar>(
						[&]() {
							return BC::traits::constexpr_if<host_mode>(
									[&](){
										return make_scalar_constant<BC::host_tag, ValueType>(alpha_mod * lv[0]);
									},[&](){
										auto tmp_scalar = make_kernel_scalar<ValueType>(stream);
										derived::scalar_multiply(stream, tmp_scalar, alpha_mod, lv);
										return tmp_scalar;
									});
						},
				BC::traits::constexpr_else([&]() { //else if rv_scalar
							return BC::traits::constexpr_if<host_mode>(
									[&](){
										return make_scalar_constant<BC::host_tag, ValueType>(alpha_mod * rv[0]);
									},[&](){
										auto tmp_scalar = make_kernel_scalar<ValueType>(stream);
										derived::scalar_multiply(stream, tmp_scalar, alpha_mod, rv);
										return tmp_scalar;
									});
						}))
				)));
	}

	template<
			class Lv,
			class Rv,
			class Alpha,
			class Beta,
			bool LvTrans,
			bool RvTrans,
			bool LvScalar,
			bool RvScalar>
	 struct contents {
		static constexpr bool lv_is_transposed = LvTrans;
		static constexpr bool rv_is_transposed = RvTrans;
		static constexpr bool lv_is_scalar_multiplied = LvScalar;
		static constexpr bool rv_is_scalar_multiplied = RvScalar;

		using left_value_type = Lv;
		using right_value_type = Rv;
		using alpha_type = Alpha;
		using beta_type = Beta;

		Lv left;
		Rv right;
		Alpha alpha;
		Beta beta;
	};

	template<int alpha_mod, int beta_mod, class Stream, class Lv, class Rv>
	static auto parse_expression(Stream stream, Lv left, Rv right) {
		/*
		 *	Strips transposition and scalar-multiplied from and left and right,
		 *	returns a 'contents' object. --
		 *
		 *	If left/right is/are transposed, calling 'dimensions(0), rows(), cols()'
		 *	will return the non-transposed dimensions/rows/cols. Ergo- you should use the original parameters
		 *	to access the shape of the returned value.
		 */
		static constexpr bool lv_scalar = blas_expression_traits<Lv>::is_scalar_multiplied::value;
		static constexpr bool rv_scalar = blas_expression_traits<Rv>::is_scalar_multiplied::value;

		using value_type = typename Lv::value_type;

		auto alpha_lv = blas_expression_traits<Lv>::get_scalar(left);
		auto alpha_rv = blas_expression_traits<Rv>::get_scalar(right);
		auto alpha_ = calculate_alpha<value_type, alpha_mod, lv_scalar, rv_scalar>(stream, alpha_lv, alpha_rv);
		auto beta_  = make_constexpr_scalar<typename expression_traits<decltype(alpha_)>::allocation_tag, beta_mod, value_type>();//blas_impl::template scalar_constant<value_type, beta_mod>();

		auto left_ = greedy_evaluate(blas_expression_traits<Lv>::remove_blas_modifiers(left), stream);
		auto right_ = greedy_evaluate(blas_expression_traits<Rv>::remove_blas_modifiers(right), stream);

		using left_t = std::decay_t<decltype(left_)>;
		using right_t = std::decay_t<decltype(right_)>;
		using alpha_t = std::decay_t<decltype(alpha_)>;
		using beta_t = std::decay_t<decltype(beta_)>;

		return contents<
				left_t,
				right_t,
				alpha_t,
				beta_t,
				blas_expression_traits<Lv>::is_transposed::value,
				blas_expression_traits<Rv>::is_transposed::value,
				blas_expression_traits<Lv>::is_scalar_multiplied::value,
				blas_expression_traits<Rv>::is_scalar_multiplied::value> { left_, right_, alpha_, beta_ };
	}

	template<class Stream, class Contents>
	static void post_parse_expression_evaluation(Stream stream, Contents contents) {
		using value_type = typename decltype(contents.alpha)::value_type;
		BC::traits::constexpr_if<(BC::tensors::exprs::expression_traits<decltype(contents.alpha)>::is_temporary::value)>(
			BC::traits::bind([&](auto alpha) {
				stream.template get_allocator_rebound<value_type>().deallocate(alpha, 1);
			}, 	contents.alpha));

		optimizer<typename Contents::left_value_type>::deallocate_temporaries(contents.left, stream);
		optimizer<typename Contents::right_value_type>::deallocate_temporaries(contents.right, stream);

	}
};


}
}
}
}

#endif
