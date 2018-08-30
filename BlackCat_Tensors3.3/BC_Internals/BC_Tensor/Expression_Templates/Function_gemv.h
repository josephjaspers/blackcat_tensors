
#ifndef EXPRESSION_BINARY_GEMV_H_
#define EXPRESSION_BINARY_GEMV_H_


#include "Array_Base.h"
#include "Expression_Base.h"
#include "BlackCat_Internal_Definitions.h"
#include "Parse_Tree_BLAS_Branch_Evaluator.h"
#include "Parse_Tree_Evaluator.h"

namespace BC {
namespace oper {
template<class ml> class gemv : public BLAS_FUNCTION {};
template<class ml> class dot;

}
namespace internal {

/*
 * a = M x K
 * b = K x N
 * c = M x N
 */


template<class lv, class rv, class mathlib>
struct binary_expression<lv, rv, oper::gemv<mathlib>>
: expression_base<binary_expression<lv, rv,  oper::gemv<mathlib>>>, BLAS_FUNCTION {

	using scalar_t  = typename lv::scalar_t;
	using mathlib_t = mathlib;

	static constexpr bool transA = det_eval<lv>::transposed;
	static constexpr bool transB = det_eval<rv>::transposed;
	static constexpr bool lvscalar_of = det_eval<lv>::scalar;
	static constexpr bool rvscalar_of = det_eval<rv>::scalar;
	static constexpr bool lv_eval = det_eval<lv>::evaluate;
	static constexpr bool rv_eval = det_eval<rv>::evaluate;

	static_assert(std::is_same<scalar_of<lv>, scalar_of<rv>>::value, "MATRIX MULTIPLICATION ONLY AVAILABLE TO SAME TYPE TENSORS (FLOAT/DOUBLE)");
	static_assert(lv::DIMS() == 2 && rv::DIMS() == 1, "GEMV DIMENSION MISMATCH, INTERNAL BUG, REPORT PLEASE");
	__BCinline__ static constexpr int DIMS() { return 1; }
	__BCinline__ static constexpr int ITERATOR() { return 0; }

	lv left;
	rv right;

	 binary_expression(lv left, rv right) : left(left), right(right) {}

	__BCinline__ int size() const { return left.rows(); }
	__BCinline__ int rows() const { return left.rows(); }
	__BCinline__ int cols() const { return 1; }
	__BCinline__ int dimension(int i) const { return i == 0 ? rows() : 1; }
	__BCinline__ int outer_dimension() const { return rows(); }

	__BCinline__ const auto inner_shape() const { return l_array<DIMS()>([&](int i) { return i == 0 ? left.rows() : 1; });}

	__BCinline__ auto _slice(int i) {
		return binary_expression<decltype(left._row(i)), decltype(right._slice(i)), oper::dot<mathlib>>(left._row(i), right._slice(i));
	}

template<class core, int alpha_mod, int beta_mod>
void eval(injection_wrapper<core, alpha_mod, beta_mod> injection_values) const {

	//get the data of the injection --> injection_wrapper simply stores the alpha/beta scalar modifiers
	auto& injection = injection_values.data();

	//evaluate the left and right branches (computes only if necessary)
	auto A = branched<mathlib>::evaluate(det_eval<lv>::get_array(left));
	auto X = branched<mathlib>::evaluate(det_eval<rv>::get_array(right));

	//initialize the alpha and beta scalars,
	scalar_t* alpha = mathlib::static_initialize((scalar_t)alpha_mod);
	scalar_t* beta  = mathlib::static_initialize((scalar_t)beta_mod);

	//get the left and right side scalar values and
	//compute the scalar values if need be
	if (lvscalar_of) {
		scalar_t* alpha_lv = det_eval<lv>::getscalar_of(left);
		mathlib::scalar_mul(alpha, alpha, alpha_lv);
	}
	if (rvscalar_of) {
		scalar_t* alpha_rv = det_eval<rv>::getscalar_of(right);
		mathlib::scalar_mul(alpha, alpha, alpha_rv);
	}

	//call matrix_mul ///for gemm we always use M, N, K regardless of transpose, but for gemv we always use pre-trans dimensions ???
	int m = A.rows();
	int n = A.cols();

	mathlib::gemv(transA,  m, n, alpha, A, A.leading_dimension(0), X, X.leading_dimension(0)/*inc_X*/, beta, injection/*Y*/, injection.leading_dimension(0)/*incy*/);

	//destroy all the temporaries
	if (lv_eval) cc(A).destroy();
	if (rv_eval) cc(X).destroy();
	mathlib::destroy(beta);
	mathlib::destroy(alpha);
}
};

}
}
//		if (transA)
//		std::cout << "A is transposed" << transA << std::endl;
//		if (transB)
//		std::cout <<"B is transposed" << transB << std::endl;
//		if (lvscalar_of)
//		std::cout << "A has scalar " <<lvscalar_of << std::endl;
//		if (rvscalar_of)
//		std::cout <<"B has scalar" << rvscalar_of << std::endl;
//		if (lv_eval)
//		std::cout << "A instant eval" <<lv_eval << std::endl;
//		if(rv_eval)
//		std::cout <<"B instant eval " << rv_eval << std::endl;

#endif /* EXPRESSION_BINARY_DOTPRODUCT_CU_ */
