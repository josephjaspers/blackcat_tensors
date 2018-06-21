///*
// * toeplitz_kernel_matrix.h
// *
// *  Created on: May 26, 2018
// *      Author: joseph
// */
//
//#ifndef TOEPLITZ_KERNEL_MATRIX_H_
//#define TOEPLITZ_KERNEL_MATRIX_H_
//
///*
// * CURRENTLY NOT USED /// TEST CODE MOST LIKELY WILL BE REMOVED IN FUTURE VERSIONS
// *
// *
// */
//
//namespace BC {
//
//template<class > struct Array;
////template<int> struct Shape;
////internal tensor type
//
//static constexpr int max(int a, int b) {
//	return a > b ? a : b;
//}
//
//template<class core_lib>
//struct CPU_Signal_Processing {
//
//	template<class T>
//	static void vv_corr_toeplitz_1d_inner(int m, int n, T alpha, T* A, int lda, T beta, T* C, int ldc, int stride = 1) {
//		//m = signal length
//		//n = kernel length
//		//A = vector of length m
//		//C = matrix of n * m
//
//		int positions = m - n + 1; //number of inner_positions of 1d corr
//
//		for (int c = 0; c < m; ++c) {
//			for (int r = 0; r < n; ++r) {
//			(C[r * ldc + c] *= beta) = A[r + c] * alpha;
//			}
//		}
//	}
//	template<class T>
//	static void mm_corr_toeplitz_1d_inner(
//								int m, int n, int k,
//								T* A, int lda,
//								T* C, int ldc, int stride = 1) {
//		//m = signal rows
//		//n = signal cols
//		//k = kernel dim (assumes square)
//		//A = Matrix of length n x m (the output for this function is transposed)
//		//C = matrix of (k^2)x((m-k+1)(n-k+1))
//
//		int r_positions = m - k + 1;
//		int c_positions = n - k + 1;
//		for (int kr = 0; kr < r_positions; ++ kr)
//			for (int kc = 0; kc < k; ++kc)
//				for (int i = 0; i < k; ++i)
//					core_lib::vec_copy(k, &A[kc * lda + kr + i], lda, &C[(kr + kc) * ldc + i * k], 1);
//	}
//	template<class T>
//	static void mm_corr_toeplitz_2d_inner(
//								int m, int n, int k,
//								T* A, int lda,
//								T* C, int ldc, int stride = 1) {
//		int r_positions = m - k + 1;
//		int c_positions = m - k + 1;
//
//		int chunk_size = k * k * r_positions;
//		int chunks = c_positions;
//
//		for (int i = 0; i < chunks; ++i) {
//			mm_corr_toeplitz_1d_inner(m, n, k, &A[i * lda], lda, &C[i * chunk_size], ldc);
//		}
//
//	}
//
//};
//
//}
//
//#endif /* TOEPLITZ_KERNEL_MATRIX_H_ */
