
#ifndef EXPRESSION_BINARY_GEMM_H_
#define EXPRESSION_BINARY_GEMM_H_


#include "Array_Base.h"
#include "BlackCat_Internal_Definitions.h"
#include "Expression_Interface.h"
#include "BLAS_Feature_Detector.h"
#include "Tree_Evaluator_Runner.h"

namespace BC {
namespace internal {
namespace oper {
template<class ml> class gemm : public BLAS_FUNCTION {};
template<class>class gemv;
}

/*
 * a = M x K
 * b = K x N
 * c = M x N
 */


template<class lv, class rv, class mathlib>
struct binary_expression<lv, rv, oper::gemm<mathlib>>
: expression_interface<binary_expression<lv, rv,  oper::gemm<mathlib>>>, BLAS_FUNCTION {


	using scalar_t  = typename lv::scalar_t;
	using mathlib_t = mathlib;

	static constexpr bool transA = blas_feature_detector<lv>::transposed;
	static constexpr bool transB = blas_feature_detector<rv>::transposed;
	static constexpr bool lvscalar_of = blas_feature_detector<lv>::scalar;
	static constexpr bool rvscalar_of = blas_feature_detector<rv>::scalar;
	static constexpr bool lv_eval = blas_feature_detector<lv>::evaluate;
	static constexpr bool rv_eval = blas_feature_detector<rv>::evaluate;

	static_assert(std::is_same<scalar_of<lv>, scalar_of<rv>>::value, "MATRIX MULTIPLICATION ONLY AVAILABLE TO SAME TYPE TENSORS (FLOAT/DOUBLE)");

	__BCinline__ static constexpr int DIMS() { return rv::DIMS(); }
	__BCinline__ static constexpr int ITERATOR() { return 0; }

	lv left;
	rv right;

	 binary_expression(lv left, rv right) : left(left), right(right) {}

	__BCinline__ const auto inner_shape() const { return l_array<DIMS()>([&](int i) { return i == 0 ? left.rows() : i == 1 ? right.cols() : 1; }); }
	__BCinline__ const auto block_shape() const { return l_array<DIMS()>([&](int i) { return i == 0 ? left.rows() : i == 1 ? size() : 1; });}

	__BCinline__ int size() const { return left.rows() * right.cols(); }
	__BCinline__ int rows() const { return left.rows(); }
	__BCinline__ int cols() const { return right.cols(); }
	__BCinline__ int dimension(int i) const { return inner_shape()[i]; }

	__BCinline__ int M() const { return left.rows();  }
	__BCinline__ int N() const { return right.cols(); }
	__BCinline__ int K() const { return left.cols();  }

	__BCinline__ auto _slice(int i) {
		return binary_expression<lv, decltype(right._slice(i)), oper::gemv<mathlib_t>>(left, right._slice(i));
	}
	__BCinline__ auto _col(int i) {
		return _slice(i);
	}

template<class core, int alpha_mod, int beta_mod>
void eval(tree::injector<core, alpha_mod, beta_mod> injection_values) const {

	//get the data of the injection --> injector simply stores the alpha/beta scalar modifiers
	auto& injection = injection_values.data();

	//evaluate the left and right branches (computes only if necessary)
	auto A = branched<mathlib>::evaluate(blas_feature_detector<lv>::get_array(left));
	auto B = branched<mathlib>::evaluate(blas_feature_detector<rv>::get_array(right));

	//get the left and right side scalar values
	scalar_t* alpha_lv = blas_feature_detector<lv>::get_scalar(left);
	scalar_t* alpha_rv = blas_feature_detector<rv>::get_scalar(right);

	//initialize the alpha and beta scalars,
	scalar_t* alpha = mathlib::static_initialize((scalar_t)alpha_mod);
	scalar_t* beta = mathlib::static_initialize((scalar_t)beta_mod);

	//compute the scalar values if need be
	if (lvscalar_of)
		mathlib::scalar_mul(alpha, alpha, alpha_lv);
	if (rvscalar_of)
		mathlib::scalar_mul(alpha, alpha, alpha_rv);

	//call matrix_mul
	mathlib::gemm(transA, transB,  M(), N(), K(), alpha, A, A.leading_dimension(0), B, B.leading_dimension(0), beta, injection, injection.leading_dimension(0));


	//destroy all the temporaries
	if (lv_eval) cc(A).destroy();
	if (rv_eval) cc(B).destroy();
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
