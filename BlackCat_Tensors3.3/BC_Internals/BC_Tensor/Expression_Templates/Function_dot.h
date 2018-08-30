/*
 * Function_dot.h
 *
 *  Created on: Aug 27, 2018
 *      Author: joseph
 */

#ifndef FUNCTION_DOT_H_
#define FUNCTION_DOT_H_

#include "Array_Base.h"
#include "Expression_Base.h"
#include "BlackCat_Internal_Definitions.h"
#include "Parse_Tree_BLAS_Branch_Evaluator.h"
#include "Parse_Tree_Evaluator.h"

namespace BC {
namespace oper {
template<class ml> class dot : public BLAS_FUNCTION {};
}
namespace internal {

/*
 * a = M x K
 * b = K x N
 * c = M x N
 */


template<class lv, class rv, class mathlib>
struct binary_expression<lv, rv, oper::dot<mathlib>>
: expression_base<binary_expression<lv, rv,  oper::dot<mathlib>>>, BLAS_FUNCTION, public Shape<0> {

	using scalar_t  = typename lv::scalar_t;
	using mathlib_t = mathlib;

	static constexpr bool transA = det_eval<lv>::transposed;
	static constexpr bool transB = det_eval<rv>::transposed;
	static constexpr bool lvscalar_of = det_eval<lv>::scalar;
	static constexpr bool rvscalar_of = det_eval<rv>::scalar;
	static constexpr bool lv_eval = det_eval<lv>::evaluate;
	static constexpr bool rv_eval = det_eval<rv>::evaluate;

	static_assert(std::is_same<scalar_of<lv>, scalar_of<rv>>::value, "MATRIX MULTIPLICATION ONLY AVAILABLE TO SAME TYPE TENSORS (FLOAT/DOUBLE)");
	static_assert(lv::DIMS() == 1 && rv::DIMS() == 1, "GEMV DIMENSION MISMATCH, INTERNAL BUG, REPORT PLEASE");
	__BCinline__ static constexpr int DIMS() { return 0; }
	__BCinline__ static constexpr int ITERATOR() { return 0; }

	lv left;
	rv right;

	 binary_expression(lv left, rv right) : left(left), right(right) {}

template<class core, int alpha_mod, int beta_mod>
void eval(injection_wrapper<core, alpha_mod, beta_mod> injection_values) const {


	//get the data of the injection --> injection_wrapper simply stores the alpha/beta scalar modifiers
	auto& injection = injection_values.data();

	//evaluate the left and right branches (computes only if necessary)
	auto X = branched<mathlib>::evaluate(det_eval<lv>::get_array(left));
	auto Y = branched<mathlib>::evaluate(det_eval<rv>::get_array(right));

	//get the left and right side scalar values

	//initialize the alpha and beta scalars,
	scalar_t* alpha = mathlib::static_initialize((scalar_t)alpha_mod);

	//compute the scalar values if need be

	//call outer product
	mathlib::dot(X.rows(), injection, X, X.leading_dimension(0), Y, Y.leading_dimension(0));

	if (lvscalar_of) {
		scalar_t* alpha_lv = det_eval<lv>::getscalar_of(left);
		mathlib::scalar_mul(injection, alpha, alpha_lv);
	}
	if (rvscalar_of) {
		scalar_t* alpha_rv = det_eval<rv>::getscalar_of(right);
		mathlib::scalar_mul(injection, alpha, alpha_rv);
	}
	if (beta_mod) {
		scalar_t* beta = mathlib::static_initialize((scalar_t)alpha_mod);
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
