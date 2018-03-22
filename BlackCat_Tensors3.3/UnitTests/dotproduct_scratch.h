/*
 * dotproduct_scratch.h
 *
 *  Created on: Mar 20, 2018
 *      Author: joseph
 */

#ifndef DOTPRODUCT_SCRATCH_H_
#define DOTPRODUCT_SCRATCH_H_

#include "../BlackCat_Tensors.h"

namespace BC{

template<class T, int row = 0, int col = 0>
struct dotproduct_impl {

template<class U, class V>
static auto foo(Matrix<T>& out, const Matrix<U>& mat1, const Matrix<V>& mat2)
{
	if (col  != mat2.cols())
		return out[row][col] +=* (mat1.row(row) % mat2[col]) && dotproduct_impl<T, row, col + 1 >::foo(out, mat1, mat2);
	else if (row != mat1.rows())
		return out[row][col] +=* (mat1.row(row) % mat2[col]) && dotproduct_impl<T, row + 1>::foo(out, mat1, mat2);
	else
		return out[row][col] +=* (mat1.row(row) % mat2[col]) && dotproduct_impl<T, row + 1, 0>::foo(out, mat1, mat2);

}
};
template<class T>
struct dotproduct_impl<T, 2, 2> {

	template<class U, class V>
static auto foo(Matrix<T>& out, const Matrix<U>& mat1, const Matrix<V>& mat2, int row = 0, int col = 0)
{
	return out[row][col];
}

};

template<class T>
auto dotproduct(const Matrix<T>& mat1, const Matrix<T>& mat2) {
	Matrix<T> out(mat1.rows(), mat2.cols());

	dotproduct_impl<T,0,0>::foo(out, mat1, mat2);
	return out;
}

}



#endif /* DOTPRODUCT_SCRATCH_H_ */

//
//std::cout << " TRANSA " << transA << std::endl;
//std::cout << " TRANSB " << transB << std::endl;
//std::cout << " ScalarA " << lv_scalar << std::endl;
//std::cout << " ScalarB " << rv_scalar << std::endl;
//std::cout << " eval_L " << lv_eval << std::endl;
//std::cout << " eval_R " << rv_eval << std::endl;

//#ifndef EXPRESSION_BINARY_DOTPRODUCT_CU_
//#define EXPRESSION_BINARY_DOTPRODUCT_CU_
//
//#include "Expression_Base.h"
//#include "Expression_Binary_Dotproduct_impl.h"
//#include "BlackCat_Internal_Definitions.h"
//#include <memory>
//
//namespace BC {
//
///*
// * a = M x K
// * b = K x N
// * c = M x N
// */
////det_Eval
//template<bool scalA, bool transA>
//struct scalar_accessor {
//	template<class T> static  auto getScalar(const T& tensor) {
//		return inferior_type<typename decltype(tensor.array)::lv, typename decltype(tensor.array)::rv>::shape(tensor.array.left, tensor.array.right);
//	}
//};
//template<>
//struct scalar_accessor<true, false> {
//	template<class T> static auto getScalar(const T& tensor) {
//		return inferior_type<typename T::lv, typename T::rv>::shape(tensor.left, tensor.right);
//	}
//};
//template<bool transA>
//struct scalar_accessor<false, transA> {
//	template<class T>  static auto getScalar(const T& tensor) {
//		return nullptr;
//	}
//};
//
//
//template<class T, bool transA, bool transB, bool scalA, bool scalB, class lv, class rv, class Mathlib>
//struct binary_expression_dotproduct : expression<T, binary_expression_dotproduct<T, transA, transB, scalA, scalB, lv, rv, Mathlib>> {
//	using scalar_type = T;
//
//
//	lv left;
//	rv right;
//
//	T* alpha = scalar_accessor<scalA, transA>::getScalar(left);
//	T* beta = scalar_accessor<scalB, transB>::getScalar(left);
//
//		struct deleter {
//			template<class param>
//			void operator () (param& p) {
//				Mathlib::destroy(p);
//			}
//		};
//
//			static constexpr bool lv_eval = det_eval<lv>::evaluate;
//			static constexpr bool rv_eval = det_eval<rv>::evaluate;
//
//
//		std::shared_ptr<scalar_type> array;
//		scalar_type* array_ptr;
//
//		__attribute__((always_inline))
//		binary_expression_dotproduct(lv left, rv right) :
//		left(left), right(right) {
//
//			std::cout << "detect scalA " << scalA << std::endl;
//			std::cout << "detect scalB " << scalB <<std::endl;
//			std::cout << "detect transA " << transA <<std::endl;
//			std::cout << "detect trasnB " << transB <<std::endl;
//
//
//			Mathlib::initialize(array_ptr,eval_size);
//			array = std::shared_ptr<scalar_type>(array_ptr, deleter());
//			eval();
//		}
//			__BCinline__ const T& operator [](int index) const {
//				return array_ptr[index];
//			}
//			__BCinline__ T& operator [](int index) {
//				return array_ptr[index];
//			}
//		int eval_size = left.rows() * right.cols();
//		__BCinline__ int size() const { return eval_size;}
//		__BCinline__ int rows() const { return left.rows();}
//		__BCinline__ int cols() const { return right.cols();}
//		__BCinline__ int rank() const { return right.rank(); }
//		__BCinline__ int LD_rows() const { return rows(); }
//		__BCinline__ int LD_cols() const { return eval_size; }
//		__BCinline__ int dimension(int i)		const { return i == 0 ? rows() : i == 1 ? cols(): 1; }
//		__BCinline__ const auto innerShape() 	const { return generateDimList(rows(), cols()); }
//		__BCinline__ const auto outerShape() 	const { return generateDimList(LD_rows(), LD_cols()); }
//
//		__BCinline__ int M() const {return left.rows(); }
//		__BCinline__ int N() const {return right.cols(); }
//		__BCinline__ int K() const {return left.cols(); }
//
//		void printDimensions() 		const { std::cout<<"[" << rows() << "][" <<cols()  <<"]" << std::endl; }
//		void printLDDimensions()	const { std::cout<<"[" << rows() << "][" <<eval_size  <<"]" << std::endl; }
//
//
//	public:
//
//		void eval() {
//
//			T* A = nullptr;
//			T* B = nullptr;
//
//
//			if (lv_eval) {
//				Mathlib::initialize(A, left.size());
//				Mathlib::copy(A, left, left.size());
//			} else {
//				A = det_eval<lv>::getArray(left);
//			}
//			if (rv_eval) {
//				Mathlib::initialize(B, right.size());
//				Mathlib::copy(B, right, right.size());
//			} else {
//				B = det_eval<rv>::getArray(right);
//			}
//
//
//				//if scalars on both sides we need to convert them into a single scalar (for it to work with blas)
//			if (scalA && scalB){
//				T* tmp;
//				Mathlib::initialize(tmp, 1);
//				Mathlib::scalarMul(tmp, alpha, beta);
//				Mathlib::MatrixMul(transA, transB, A, B, array_ptr, M(), N(), K(), tmp, nullptr, left.LD_rows(), right.LD_rows(), rows());
//				Mathlib::destroy(tmp);
//
//			} else if (scalA)
//				 Mathlib::MatrixMul(transA, transB, A, B, array_ptr, M(), N(), K(), beta, nullptr, left.LD_rows(), right.LD_rows(), rows());
//			 else if (scalB)
//				 Mathlib::MatrixMul(transA, transB, A, B, array_ptr, M(), N(), K(), alpha, nullptr, left.LD_rows(), right.LD_rows(), rows());
//			 else
//				 Mathlib::MatrixMul(transA, transB, A, B, array_ptr, M(), N(), K(), nullptr, nullptr, left.LD_rows(), right.LD_rows(), rows());
//
//			if (lv_eval)
//				Mathlib::destroy(A);
//			if (rv_eval)
//				Mathlib::destroy(B);
//
//		}
//
//};
//}
//#endif



//	template<class param_deriv>
//	struct dp_impl {
//		//Determines the return type dotproducts (including pointwise scalar operations)
//		using param_functor_type = typename Tensor_Operations<param_deriv>::functor_type;
//		static constexpr bool lv_scalar = derived::RANK() == 0;
//		static constexpr bool rv_scalar = param_deriv::RANK() == 0;
//		static constexpr bool scalar_mul = lv_scalar || rv_scalar;
//
//		static constexpr bool evaluate_to_vector = derived::RANK() == 2 && param_deriv::RANK() == 1;
//		static constexpr bool evaluate_to_matrix = derived::RANK() == 2 && param_deriv::RANK() == 2;
//		static constexpr bool evaluate_to_mat_vv = derived::RANK() == 1 && param_deriv::RANK() == 1;
//		static constexpr bool evaluate_to_dominant = derived::RANK() == 0 || param_deriv::RANK() == 0;
//
//		static constexpr bool short_params = lv_scalar || rv_scalar;
//
//		using mulType      =   typename MTF::IF_ELSE<lv_scalar || rv_scalar,
//									typename MTF::IF_ELSE<lv_scalar,
//										typename MTF::expression_substitution<binary_expression_scalar_L<scalar_type, mul, functor_type, param_functor_type>, param_deriv>::type,
//										typename MTF::expression_substitution<binary_expression_scalar_R<scalar_type, mul, functor_type, param_functor_type>, derived	 >::type
//									>::type,
//								void>::type;
//		using vecType = typename MTF::expression_substitution<binary_expression_dotproduct<scalar_type, functor_type, param_functor_type, math_library>,param_deriv>::type;
//		using matType = typename MTF::expression_substitution<binary_expression_dotproduct<scalar_type, functor_type, param_functor_type, math_library>,param_deriv>::type;
//		using outerType = typename MTF::expression_substitution<binary_expression_dotproduct<scalar_type, functor_type, param_functor_type, math_library>,
//									Matrix<functor_type, math_library>>::type;
//
//		using type 			= 	typename MTF::IF_ELSE<evaluate_to_vector, vecType,
//									typename MTF::IF_ELSE<evaluate_to_matrix, matType,
//										typename MTF::IF_ELSE<evaluate_to_mat_vv, outerType,
//											mulType
//										>::type
//									>::type
//								>::type;
//
//		using expression_type = type;
//
//	};


//	//Pointwise multiplication of Scalar and Tensor -- is auto-detected by expression templates when in conjunction to dotproduct
//	template<class pDeriv, class voider = typename std::enable_if<dp_impl<pDeriv>::short_params || dp_impl<pDeriv>::short_params>::type>
//	typename dp_impl<pDeriv>::type operator *(const Tensor_Operations<pDeriv>& param) const {
//			return typename dp_impl<pDeriv>::type(this->data(), param.data());
//	}
	/*
	 * a = M x K
	 * b = K x N
	 * c = M x N
	 */
	//Dot product implementation
//	template<class pDeriv,
//		class voider = typename std::enable_if<!dp_impl<pDeriv>::short_params && !dp_impl<pDeriv>::short_params>::type, int foo = 0>
//	typename dp_impl<pDeriv>::type operator *(const Tensor_Operations<pDeriv>& param) const {
//		assert_same_ml(param);
//		return typename dp_impl<pDeriv>::type(this->data(), param.data());
//	}
	//This allows you to do the operator ** as a point-wise multiplication operation

	//point-wise multiplication overload (represented by **)


