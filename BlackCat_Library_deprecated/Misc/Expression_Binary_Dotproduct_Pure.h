///*
// * Expression_Binary_Dotproduct_Pure.cu
// *
// *  Created on: Mar 6, 2018
// *      Author: joseph
// */
//
//#ifndef EXPRESSION_BINARY_DOTPRODUCT_PURE_CU_
//#define EXPRESSION_BINARY_DOTPRODUCT_PURE_CU_
//
//namespace BC {
//
//template<class T, class lv, class rv>
//struct binary_expression_dotproduct_pure {
//
//	lv left;
//	rv right;
//
//	binary_expression_dotproduct_pure(lv left_, rv right_) : left(left_), right(right_) {}
//
//	T operator [] (int i) const {
//		int l_row = i % left.rows();
//		int l_base = l_row;
//
//		int r_col = (int)(i / left.rows());
//		int r_base = r_col * right.LD_rows();
//
//		T sum = 0;
//		for (int i = 0; i < left.cols(); ++i) {
//			sum += left[l_base + i * left.LD_rows()] * right[r_base + i];
//		}
//		return sum;
//	}
//};
//
//
//}
//
//
//
//#endif /* EXPRESSION_BINARY_DOTPRODUCT_PURE_CU_ */


/*
 * Expression_Binary_Dotproduct.cu
 *
 *  Created on: Jan 9, 2018
 *      Author: joseph
 */


//
//template<class T, class lv, class rv, class ml>
//struct binary_expression_dotproduct {
//
//	lv left;
//	rv right;
//
//	binary_expression_dotproduct(lv left_, rv right_) : left(left_), right(right_) {}
//
//	__BCinline__ T operator [] (int i) const {
//		int l_row = i % left.rows();
//		int l_base = l_row;
//
//		int r_col = (int)(i / left.rows());
//		int r_base = r_col * right.LD_rows();
//
//		T sum = 0;
//		for (int i = 0; i < left.cols(); ++i) {
//			sum += left[l_base + i * left.LD_rows()] * right[r_base + i];
//		}
//		return sum;
//	}
//		__BCinline__ int size() const { return left.rows() * right.cols();}
//		__BCinline__ int rows() const { return left.rows();}
//		__BCinline__ int cols() const { return right.cols();}
//		__BCinline__ int rank() const { return right.rank(); }
//		__BCinline__ int LD_rows() const { return left.rows(); }
//		__BCinline__ int LD_cols() const { return right.cols(); }
//		__BCinline__ int dimension(int i)		const { return i== 0 ? left.rows() : i == 1 ? right.cols() : 1; }
//		void printDimensions() const {}
//		void printLDDimensions() const {}
////		__BCinline__ const auto innerShape() 	const { return is; }
////		__BCinline__ const auto outerShape() 	const { return os; }
//};

//				Some printouts for debugging
//
//		std::cout << "dotproduct stats --------------------------------------------------------------------------" << std::endl;
//				std::cout << " m n k = " << M << "  "<< N << " " << K << std::endl;
//
//				if (lv_needs_to_be_evaluated) {
//					if (self_eval<lv>::conditioanl) {
//						std::cout << " lv self eval " << std::endl;
//					}
//					std::cout << " lv was evaluated " << std::endl;
//				}
//				if (rv_needs_to_be_evaluated) {
//					if (self_eval<rv>::conditioanl) {
//						std::cout << " rv self eval " << std::endl;
//					}
//					std::cout << " lv was evaluated " << std::endl;
//				}
//				if (transA) {
//					std::cout << " A - fast trans " << std::endl;
//				}
//				if (transB) {
//					std::cout << " B - fast trans " << std::endl;
//				}
//				if (evaluate<lv>::scalar) {
//					std::cout << " lv scalar detected " << std::endl;
//				}
//				if (evaluate<rv>::scalar) {
//					std::cout << " rv scalar detected " << std::endl;
//				}
//				if (scal_A && scal_B)
//				std::cout << "scalars = " << *scal_A <<  " " << *scal_B << std::endl;
//				std::cout << " --------------------------------------------------------------------------" << std::endl;



