
#ifndef EXPRESSION_BINARY_DOTPRODUCT_CU_
#define EXPRESSION_BINARY_DOTPRODUCT_CU_

#include "Core_Base.h"

#include "Expression_Base.h"

#include "BLAS_Expression_Evaluator.h"
#include "BLAS_Injection_Runner.h"

#include "BlackCat_Internal_Definitions.h"

namespace BC {
namespace oper {
template<class ml> class dotproduct : public BLAS_FUNCTION {};
}
namespace internal {

/*
 * a = M x K
 * b = K x N
 * c = M x N
 */


template<class lv, class rv, class mathlib>
struct binary_expression<lv, rv, oper::dotproduct<mathlib>> : expression_base<binary_expression<lv, rv,  oper::dotproduct<mathlib>>> {

	using scalar_type = _scalar<lv>;

	__BCinline__ static constexpr int DIMS() { return rv::DIMS(); }
	__BCinline__ static constexpr int ITERATOR() { return 0; }

	static constexpr bool transA = det_eval<lv>::transposed;
	static constexpr bool transB = det_eval<rv>::transposed;
	static constexpr bool lv_scalar = det_eval<lv>::scalar;
	static constexpr bool rv_scalar = det_eval<rv>::scalar;
	static constexpr bool lv_eval = det_eval<lv>::evaluate;
	static constexpr bool rv_eval = det_eval<rv>::evaluate;

	lv left;
	rv right;

	 binary_expression(lv left, rv right) : left(left), right(right) {}
	 operator Core<tensor_of_t<DIMS(), scalar_type, mathlib>> () const {
		auto tc = Core<tensor_of_t<DIMS(), scalar_type, mathlib>>(this->inner_shape());
		this->eval(tc);
	 }

	__BCinline__ const auto inner_shape() const { return l_array([&](int i) { return i == 0 ? left.rows() : i == 1 ? right.cols() : 1; }); }
	__BCinline__ const auto outer_shape() const { return l_array([&](int i) { return i == 0 ? left.rows() : i == 1 ? right.cols() * left.rows() : 1; }); }

	__BCinline__ int M() const { return left.rows();  }
	__BCinline__ int N() const { return right.cols(); }
	__BCinline__ int K() const { return left.cols();  }

public:

template<class core>
void eval(core injection) const {
//
	using lv_A = decltype(branched<mathlib, true>::evaluate(det_eval<lv>::get_array(left)));
	using rv_B = decltype(branched<mathlib, true>::evaluate(det_eval<rv>::get_array(right)));
	lv_A A = branched<mathlib, true>::evaluate(det_eval<lv>::get_array(left));
	rv_B B = branched<mathlib, true>::evaluate(det_eval<rv>::get_array(right));

	scalar_type* alpha = nullptr;
	scalar_type* alpha2 = nullptr;

	alpha = det_eval<lv>::get_scalar(left);
	alpha2 = det_eval<rv>::get_scalar(right);

	if (lv_scalar && rv_scalar) {
		//in case C = A*a * (B*b) -- multiply both scalars
		scalar_type* tmp;
		mathlib::initialize(tmp, 1);
		mathlib::scalarMul(tmp, alpha, alpha2);
		mathlib::gemm(transA, transB, A, B, injection, M(), N(), K(), tmp, nullptr, left.ld1(), right.ld1(), injection.ld1());
		mathlib::destroy(tmp);

	} else if (rv_scalar)
	mathlib::gemm(transA, transB, A, B, injection, M(), N(), K(), alpha2, nullptr, left.ld1(), right.ld1(), injection.ld1());
	else if (lv_scalar)
	mathlib::gemm(transA, transB, A, B, injection, M(), N(), K(), alpha, nullptr, left.ld1(), right.ld1(), injection.ld1());
	else
	mathlib::gemm(transA, transB, A, B, injection, M(), N(), K(), nullptr, nullptr, left.ld1(), right.ld1(), injection.ld1());

	if (lv_eval) {
		cc(A).destroy();
	}
	if (rv_eval) {
		cc(B).destroy();
	}
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

//		scalar_type* A = nullptr;
//		scalar_type* B = nullptr;
//				if (lv_eval) {
//					mathlib::initialize(A, left.size());
//					mathlib::copy(A, left, left.size());
//				} else {
//					A = det_eval<lv>::get_array(left);
//				}
//				if (rv_eval) {
//					mathlib::initialize(B, right.size());
//					mathlib::copy(B, right, right.size());
//				} else {
//					B = det_eval<rv>::get_array(right);
//				}
//


#endif /* EXPRESSION_BINARY_DOTPRODUCT_CU_ */
