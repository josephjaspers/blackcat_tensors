/*
 * Function_dot.h
 *
 *  Created on: Aug 27, 2018
 *      Author: joseph
 */

#ifndef FUNCTION_DOT_H_
#define FUNCTION_DOT_H_

#include "Expression_Base.h"
#include "Internal_BLAS_Feature_Detector.h"
#include "Tree_Evaluator_Runner.h"

namespace BC {
namespace internal {
namespace oper {
template<class ml> class dot;
}
/*
 * a = M x K
 * b = K x N
 * c = M x N
 */


template<class lv, class rv, class mathlib>
struct Binary_Expression<lv, rv, oper::dot<mathlib>>
: Expression_Base<Binary_Expression<lv, rv,  oper::dot<mathlib>>>, BLAS_FUNCTION, Shape<0> {

	using scalar_t  = typename lv::scalar_t;
	using mathlib_t = mathlib;

	static constexpr bool transA = blas_feature_detector<lv>::transposed;
	static constexpr bool transB = blas_feature_detector<rv>::transposed;
	static constexpr bool lv_scalar = blas_feature_detector<lv>::scalar;
	static constexpr bool rv_scalar = blas_feature_detector<rv>::scalar;
	static constexpr bool lv_eval = blas_feature_detector<lv>::evaluate;
	static constexpr bool rv_eval = blas_feature_detector<rv>::evaluate;

	static_assert(std::is_same<scalar_of<lv>, scalar_of<rv>>::value, "MATRIX MULTIPLICATION ONLY AVAILABLE TO SAME TYPE TENSORS (FLOAT/DOUBLE)");
	static_assert(lv::DIMS() == 1 && (rv::DIMS() == 1 || rv::DIMS() ==0), "DOT DIMENSION MISMATCH, INTERNAL BUG, REPORT PLEASE");
	__BCinline__ static constexpr int DIMS() { return 0; }
	__BCinline__ static constexpr int ITERATOR() { return 0; }

	lv left;
	rv right;

	 Binary_Expression(lv left, rv right) : left(left), right(right) {}

template<class core, int alpha_mod, int beta_mod>
void eval(tree::injector<core, alpha_mod, beta_mod> injection_values) const {


	//get the data of the injection --> injector simply stores the alpha/beta scalar modifiers
	auto& injection = injection_values.data();

	//evaluate the left and right branches (computes only if necessary)
	auto X = CacheEvaluator<mathlib>::evaluate(blas_feature_detector<lv>::get_array(left));
	auto Y = CacheEvaluator<mathlib>::evaluate(blas_feature_detector<rv>::get_array(right));

	//initialize the alpha and beta scalars,
	auto alpha = mathlib::static_initialize((scalar_t)alpha_mod);

	//call outer product
	mathlib::dot(X.rows(), injection, X, X.leading_dimension(0), Y, Y.leading_dimension(0));

	if (lv_scalar) {
		auto alpha_lv = blas_feature_detector<lv>::get_scalar(left);
		mathlib::scalar_mul(injection, alpha, alpha_lv);
	}
	if (rv_scalar) {
		auto alpha_rv = blas_feature_detector<rv>::get_scalar(right);
		mathlib::scalar_mul(injection, alpha, alpha_rv);
	}
	if (beta_mod) {
		auto beta = mathlib::static_initialize((scalar_t)alpha_mod);
		mathlib::scalar_mul(alpha, alpha, beta);
		mathlib::destroy(beta);
	}


	//destroy all the temporaries
	if (lv_eval) cc(X).destroy();
	if (rv_eval) cc(Y).destroy();
	mathlib::destroy(alpha);
}
};

}
}



#endif /* FUNCTION_DOT_H_ */
