/*
 * Expression_Binary_Dotproduct.cu
 *
 *  Created on: Jan 9, 2018
 *      Author: joseph
 */

#ifndef EXPRESSION_BINARY_DOTPRODUCT_CU_
#define EXPRESSION_BINARY_DOTPRODUCT_CU_


#include "Expression_Binary_Dotproduct_impl.h"
#include "Expression_Binary_Dotproduct_impl2.h"

#include "../BC_MetaTemplateFunctions/Adhoc.h"

namespace BC {

/*
 * a = M x K
 * b = K x N
 * c = M x N
 */

//The evaluation of dot_products are tightly coupled with the unary expression - transpose, and binary expression scalar L and scalar R.

template<class T, class lv, class rv, class Mathlibrary>
struct binary_expression_dotproduct : expression<T, binary_expression_dotproduct<T, lv, rv, Mathlibrary>> {
	using functor_type = typename MTF::determine_functor<T>::type;

	static constexpr int ALPHA_DEFAULT  = 1;
	static constexpr int BETA_DEFAULT   = 0;

	const lv& left;
	const rv& right;

	const int M = 0;
	const int N = 0;
	const int K = 0;
	const int LDA = 0;
	const int LDB = 0;
	const int LDC = 0;

	static constexpr bool lv_needs_to_be_evaluated = evaluate<lv>::conditional;
	static constexpr bool rv_needs_to_be_evaluated = evaluate<rv>::conditional && (!evaluate<rv>::scalar); //ensure its not an operation by scalar
	static constexpr bool transA = evaluate<lv>::transpose;
	static constexpr bool transB = evaluate<rv>::transpose;

	using l_eval_array = typename MTF::determine_scalar<lv>::type;
	using r_eval_array = typename MTF::determine_scalar<rv>::type;

	const int lv_size = M * N;
	const int rv_size = N * K;
	const int eval_size = M * K ;

	__attribute__((always_inline))
	binary_expression_dotproduct(const lv& left, const rv& right, int M, int N, int K, int ldM = 0, int ldN = 0, int ldK = 0) :
	M(M), N(N), K(K),
	LDA(ldM == 0 ? M : ldM), LDB(ldN == 0 ? K : ldN), LDC(ldK == 0 ? M : ldK),
	left(left), right(right) { eval(); }

	functor_type array; //will always be pointer //C

public:

	void eval() {
		auto A_not_eval = evaluate<lv>::getArray(left);
		auto B_not_eval = evaluate<rv>::getArray(right);
		auto scal_A = evaluate<lv>::getScalar(left);
		auto scal_B = evaluate<rv>::getScalar(right);

		T* A;
		T* B;

		if (lv_needs_to_be_evaluated) {
			Mathlibrary::initialize(A, lv_size);
			Mathlibrary::copy(A, A_not_eval, lv_size);
		}
			else { A = A_not_eval; }

		if (rv_needs_to_be_evaluated) {
			Mathlibrary::initialize(B, rv_size);
			Mathlibrary::copy(B, B_not_eval, rv_size);
		}
			else { B = B_not_eval; }

		Mathlibrary::initialize(array, eval_size);
		Mathlibrary::MatrixMul(transA, transB, A, B, array, M, N, K, scal_A, scal_B, LDA, LDB, LDC);


		if (lv_needs_to_be_evaluated) { Mathlibrary::destroy(A); }
		if (rv_needs_to_be_evaluated) { Mathlibrary::destroy(B); }
//
//		if (lv_needs_to_be_evaluated) {
//			std::cout << " lv was evaluated " << std::endl;
//		}
//		if (rv_needs_to_be_evaluated) {
//			std::cout << " lv was evaluated " << std::endl;
//		}
//		if (transA) {
//			std::cout << " A - fast trans " << std::endl;
//		}
//		if (transB) {
//			std::cout << " B - fast trans " << std::endl;
//		}
//		if (evaluate<lv>::scalar) {
//			std::cout << " lv scalar detected " << std::endl;
//		}
//		if (evaluate<rv>::scalar) {
//			std::cout << " rv scalar detected " << std::endl;
//		}

	}

		__attribute__((always_inline))  __BC_gcpu__
		 const T operator [](int index) const {
			return array[index];
		}
		__attribute__((always_inline))  __BC_gcpu__
			T operator [](int index) {
			return array[index];
		}
	};
}


//
//template<class T, class lv, class rv, class Mathlibrary>
//struct binary_expression_dotproduct : expression<T, binary_expression_dotproduct<T, lv, rv, Mathlibrary>> {
//	using functor_type = typename MTF::determine_functor<T>::type;
//
//	const lv& left;
//	const rv& right;
//
//	const int M = 0;
//	const int N = 0;
//	const int K = 0;
//	const int LDA = 0;
//	const int LDB = 0;
//	const int LDC = 0;
//
//	static constexpr bool lv_needs_to_be_evaluated = needs_evaluation<lv>::conditional;
//	static constexpr bool rv_needs_to_be_evaluated = needs_evaluation<rv>::conditional;
//	static constexpr bool transA = 		 extract_dp<lv>::fast_trans;
//	static constexpr bool transB = extract_dp_Bside<rv>::fast_trans;
//
//	using l_eval_array = typename MTF::determine_scalar<lv>::type;
//	using r_eval_array = typename MTF::determine_scalar<rv>::type;
//
//	const int lv_size = M * N;
//	const int rv_size = N * K;
//	const int eval_size = M * K ;
//
//
//	__attribute__((always_inline))
//	binary_expression_dotproduct(const lv& left, const rv& right, int M, int N, int K, int ldM = 0, int ldN = 0, int ldK = 0) :
//	M(M), N(N), K(K),
//	LDA(ldM == 0 ? M : ldM), LDB(ldN == 0 ? K : ldK), LDC(ldK == 0 ? M : ldK),
//	left(left), right(right) {
//		eval();
//	}
//
//	functor_type array; //will always be pointer
//
//public:
//
//	void eval() {
//		const T*  scal_A = extract_dp<lv>::getScalar(left);
//		const T*  A = 	extract_dp<lv>::getArray(left, lv_size, Mathlibrary());
//		const T*  scal_B = extract_dp_Bside<rv>::getScalar(right); //nullptr (always) -- I thought blas was y = Aa * Bb... buts its y = yC + aA * b;
//		const T*  B = extract_dp_Bside<rv>::getArray(right, rv_size, Mathlibrary());
//
////		std::cout << " rv evaluation should be false  " << rv_needs_to_be_evaluated << std::endl;
////		std::cout << " rv evaluation fast trans_should be true  " << transB << std::endl;
//
//		Mathlibrary::initialize(array, eval_size);
//		Mathlibrary::MatrixMul(transA, transB, A, B, array, M, N, K, scal_A, scal_B, LDA, LDB, LDC);
//
//		if /*constexpr*/ (lv_needs_to_be_evaluated) {
//			Mathlibrary::destroy(const_cast<T*>(A));
//		}
//		if /*constexpr*/ (rv_needs_to_be_evaluated) {
//			Mathlibrary::destroy(const_cast<T*>(B));
//		}
//	}
//
//		__attribute__((always_inline))  __BC_gcpu__
//		 const T operator [](int index) const {
//			return array[index];
//		}
//		__attribute__((always_inline))  __BC_gcpu__
//			T operator [](int index) {
//			return array[index];
//		}
//	};
//}

//template<class T, class lv, class rv, class Mathlibrary, int m = 0, int n = 0, int k = 0, int ldA = m, int ldB = k, int ldC = m, bool transAlpha = false, bool transBeta = false> //will deterministic for cuda/cpu
//struct binary_expression_dotproduct : expression<T, binary_expression_dotproduct<T, lv, rv, Mathlibrary, m, n, k, ldA, ldB, ldC>> {
//	using functor_type = typename MTF::determine_functor<T>::type;
//
//	const lv& left;
//	const rv& right;
//
//	const int M = m;
//	const int N = n;
//	const int K = k;
//	const int LDA = ldA;
//	const int LDB = ldB;
//	const int LDC = ldC;
//
//	static constexpr bool lv_needs_to_be_evaluated = needs_evaluation<lv>::conditional;
//	static constexpr bool rv_needs_to_be_evaluated = needs_evaluation<rv>::conditional;
//	static constexpr bool transA = 		 extract_dp<lv>::fast_trans || transAlpha;
//	static constexpr bool transB = extract_dp_Bside<rv>::fast_trans || transBeta;
//
//	using l_eval_array = typename MTF::determine_scalar<lv>::type;
//	using r_eval_array = typename MTF::determine_scalar<rv>::type;
//
//	static constexpr int lv_size = m * n;
//	static constexpr int rv_size = n * k;
//	static constexpr int eval_size = m * k ;
//
//	__attribute__((always_inline))
//	binary_expression_dotproduct(const lv& left, const rv& right) :
//			left(left), right(right) {
//		eval();
//	}
//	__attribute__((always_inline))
//	binary_expression_dotproduct(const lv& left, const rv& right, int M, int N, int K, int ldM = 0, int ldN = 0, int ldK = 0) :
//	M(M), N(N), K(K),
//	LDA(ldM == 0 ? M : ldM), LDB(ldN == 0 ? K : ldK), LDC(ldK == 0 ? M : ldK),
//	left(left), right(right) {
//		eval();
//	}
//
//	functor_type array; //will always be pointer
//
//public:
//
//	void eval() {
//		const T*  scal_A = extract_dp<lv>::getScalar(left);
//		const T*  A = 	extract_dp<lv>::getArray(left, lv_size, Mathlibrary());
//		const T*  scal_B = extract_dp_Bside<rv>::getScalar(right); //nullptr (always) -- I thought blas was y = Aa * Bb... buts its y = yC + aA * b;
//		const T*  B = extract_dp_Bside<rv>::getArray(right, rv_size, Mathlibrary());
//		Mathlibrary::initialize(array, eval_size);
//		Mathlibrary::MatrixMul(transA, transB, A, B, array, M, N, K, scal_A, scal_B, LDA, LDB, LDC);
//
//		if /*constexpr*/ (lv_needs_to_be_evaluated) {
//			Mathlibrary::destroy(const_cast<T*>(A));
//		}
//		if /*constexpr*/ (rv_needs_to_be_evaluated) {
//			Mathlibrary::destroy(const_cast<T*>(B));
//		}
//	}
//
//		__attribute__((always_inline))  __BC_gcpu__
//		 const T operator [](int index) const {
//			return array[index];
//		}
//		__attribute__((always_inline))  __BC_gcpu__
//			T operator [](int index) {
//			return array[index];
//		}
//	};
//}



#endif /* EXPRESSION_BINARY_DOTPRODUCT_CU_ */
