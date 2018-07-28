
#ifndef EXPRESSION_BINARY_GER_H_
#define EXPRESSION_BINARY_GER_H_


#include "Array_Base.h"
#include "Expression_Base.h"
#include "BlackCat_Internal_Definitions.h"
#include "Parse_Tree_BLAS_Branch_Evaluator.h"
#include "Parse_Tree_Evaluator.h"

namespace BC {
namespace oper {
template<class ml> class ger : public BLAS_FUNCTION {};
}
namespace internal {

/*
 * a = M x K
 * b = K x N
 * c = M x N
 */


template<class lv, class rv, class mathlib>
struct binary_expression<lv, rv, oper::ger<mathlib>>
: expression_base<binary_expression<lv, rv,  oper::ger<mathlib>>>, BLAS_FUNCTION {

	using scalar_type = _scalar<lv>;


	static constexpr bool transA = det_eval<lv>::transposed;
	static constexpr bool transB = det_eval<rv>::transposed;
	static constexpr bool lv_scalar = det_eval<lv>::scalar;
	static constexpr bool rv_scalar = det_eval<rv>::scalar;
	static constexpr bool lv_eval = det_eval<lv>::evaluate;
	static constexpr bool rv_eval = det_eval<rv>::evaluate;

	static_assert(std::is_same<_scalar<lv>, _scalar<rv>>::value, "MATRIX MULTIPLICATION ONLY AVAILABLE TO SAME TYPE TENSORS (FLOAT/DOUBLE)");
	static_assert(lv::DIMS() == 1 && rv::DIMS() == 1 && transB, "GER DIMENSION MISMATCH, INTERNAL BUG, REPORT PLEASE");
	__BCinline__ static constexpr int DIMS() { return 2; }
	__BCinline__ static constexpr int ITERATOR() { return 0; }

	lv left;
	rv right;

	 binary_expression(lv left, rv right) : left(left), right(right) {}
	__BCinline__ int size() const { return left.size() * right.size(); }
		__BCinline__ int rows() const { return left.rows(); }
		__BCinline__ int cols() const { return right.cols(); }
		__BCinline__ int dimension(int i) const { return i == 0 ? rows() : i == 1 ? cols() : 1; }
		__BCinline__ int outer_dimension() const { return rows(); }

	__BCinline__ const auto inner_shape() const { return l_array<DIMS()>([&](int i) { return i == 0 ? left.rows() : i == 1 ? right.rows() : 1; });}

	__BCinline__ int M() const { return left.rows();  }
	__BCinline__ int N() const { return right.rows(); }


template<class core, int alpha_mod, int beta_mod>
void eval(injection_wrapper<core, alpha_mod, beta_mod> injection_values) const {

	//get the data of the injection --> injection_wrapper simply stores the alpha/beta scalar modifiers
	auto& injection = injection_values.data();

	//evaluate the left and right branches (computes only if necessary)
	auto A = branched<mathlib>::evaluate(det_eval<lv>::get_array(left));
	auto B = branched<mathlib>::evaluate(det_eval<rv>::get_array(right));

	//get the left and right side scalar values
	scalar_type* alpha_lv = det_eval<lv>::get_scalar(left);
	scalar_type* alpha_rv = det_eval<rv>::get_scalar(right);

	//initialize the alpha and beta scalars,
	scalar_type* alpha = mathlib::static_initialize((scalar_type)alpha_mod);

	//compute the scalar values if need be
	if (lv_scalar)
		mathlib::scalar_mul(alpha, alpha, alpha_lv);
	if (rv_scalar)
		mathlib::scalar_mul(alpha, alpha, alpha_rv);

	//call outer product
	mathlib::ger(M(), N(), alpha, A, 1, B, 1, injection, injection.leading_dimension(0));


	//destroy all the temporaries
	if (lv_eval) cc(A).destroy();
	if (rv_eval) cc(B).destroy();
	mathlib::destroy(alpha);
}
};

}
}
//		if (transA)
//		std::cout << "A is transposed" << transA << std::endl;
//		if (transB)
//		std::cout <<"B is transposed" << transB << std::endl;
//		if (lv_scalar)
//		std::cout << "A has scalar " <<lv_scalar << std::endl;
//		if (rv_scalar)
//		std::cout <<"B has scalar" << rv_scalar << std::endl;
//		if (lv_eval)
//		std::cout << "A instant eval" <<lv_eval << std::endl;
//		if(rv_eval)
//		std::cout <<"B instant eval " << rv_eval << std::endl;

#endif /* EXPRESSION_BINARY_DOTPRODUCT_CU_ */
