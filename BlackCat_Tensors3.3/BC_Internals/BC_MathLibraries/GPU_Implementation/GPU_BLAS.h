/*
 * GPU_BLAS.h
 *
 *  Created on: May 6, 2018
 *      Author: joseph
 */

#ifndef GPU_BLAS_H_
#define GPU_BLAS_H_

namespace BC{

namespace constants {
static float* static_initialize(int sz, float value) {
	float* t;
	cudaMallocManaged((void**) &t, sizeof(float));
	cudaMemcpy(t, &value, sizeof(float), cudaMemcpyHostToDevice);
	return t;
}


static float* const s1 = static_initialize(1, 1);
static float* const s0 = static_initialize(1, 0);
static float* const sn1 = static_initialize(1, -1);

}

template<class core_lib>
struct GPU_BLAS {

	static void gemm(bool transA, bool transB, int m, int n, int k,
			const float* alpha, const float* A, int lda,
								const float* B, int ldb,
			const float* beta, 		  float* C, int ldc) {
		auto TRANS_A = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
		auto TRANS_B = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

		cublasHandle_t handle;
		cublasCreate(&handle);
		cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

//		if (lda == 0 ) lda = m;
//		if (ldb == 0 ) ldb = n;
//		if (ldc == 0 ) ldc = m;
//		const float *const alpha = scalarA ? const_cast<float*>(scalarA) : constants::s1;
//		const float *const beta = scalarC ?  const_cast<float*>(scalarC) : constants::s0;

		cublasSgemm(handle, TRANS_A, TRANS_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
		cudaDeviceSynchronize();
		cublasDestroy(handle);
	}
};


}



#endif /* GPU_BLAS_H_ */
