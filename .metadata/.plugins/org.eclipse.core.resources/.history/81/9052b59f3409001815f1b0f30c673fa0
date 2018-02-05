
#ifndef MATHEMATICS_Managed_GPU_H_
#define MATHEMATICS_Managed_GPU_H_
#ifdef  BLACKCAT_GPU_ENABLED

#include <cmath>
#include <iostream>
#include <string>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "BC_PrintFunctions.h"
#include "Mathematics_GPU_impl.cu"
#include "BC_PrintFunctions.h"

#include <cublas_v2.h>



namespace BC {


class mGPU {


public:
	/*
	 * T -- An object (Generally an array) with a functional [] operator
	 * J -- Either a value on the stack to be assigned or another array similar to above.
	 *
	 * J -- may also be a single value passed by pointer, certains methods are overloaded to handle these instance
	 *
	 */

	static int blocks(int size) {
		return (size + CUDA_BASE_THREADS - 1) / CUDA_BASE_THREADS;
	}
	static int threads() {
		return CUDA_BASE_THREADS;
	}
	//destructor - no destructor are for controlling destruction of the pointer
	template<typename T>
	static void initialize(T*& t, int sz) {
//		cudaMalloc((void**)&t, sizeof(T) * sz);
		cudaMallocManaged((void**)&t, sizeof(T) * sz);

	}
	template<typename T>
	static void	 unified_initialize(T*& t, int sz) {
		cudaMallocManaged((void**)&t, sizeof(T) * sz);
	}

	template<class T, class U>
	static void copy(T t, U u, int sz) {
		gpu_impl::copy<<<blocks(sz),threads()>>>(t, u, sz);
		cudaDeviceSynchronize();
	}
	template<class T, class U>
	static void copy(T* t, U u, int sz) {
		gpu_impl::copy<<<blocks(sz),threads()>>>(t, u, sz);
		cudaDeviceSynchronize();
	}
	template<class T>
	static void copy(T t, int sz) {
		gpu_impl::eval<<<blocks(sz),threads()>>>(t, sz);
		cudaDeviceSynchronize();
	}

	template<typename T>
	static void destroy(T* t) {
		cudaFree(t);
	}
	template<typename T>
	static void destroy(T t) {
		throw std::invalid_argument("destruction on class object");
	}
	template<typename T, typename J>
	static void fill(T& t, const J j, int sz) {
		gpu_impl::fill<<<blocks(sz),threads()>>>(t, j, sz);
		cudaDeviceSynchronize();
	}

	template<typename T, typename J>
	static void set_heap(T *t, J *j) {
		gpu_impl::set_heap<<<1,1>>>(t, j);
		cudaDeviceSynchronize();
	}
	template<typename T, typename J>
	static void set_stack(T *t, J j) {
		gpu_impl::set_stack<<<1,1>>>(t, j);
		cudaDeviceSynchronize();
	}


	template<typename T, typename J>
	static void fill(T& t, const J* j, int sz) {
		gpu_impl::fill<<<blocks(sz),threads()>>>(t, j, sz);
		cudaDeviceSynchronize();
	}
	template<typename T>
	static void zero(T& t, int sz) {
		gpu_impl::zero<<<blocks(sz),threads()>>>(t, sz);
		cudaDeviceSynchronize();
	}

	template<typename T, class J>
	static void randomize(T t, J lower_bound, J upper_bound, int sz) {
		gpu_impl::randomize<<<blocks(sz),threads()>>>(t, lower_bound, upper_bound, sz, rand());
		cudaDeviceSynchronize();
	}

	template<typename T>
	static void createScalarOne(T* non_init_ptr) {
		cudaMalloc((void**)&non_init_ptr, sizeof(float));
		gpu_impl::scalarONE<<<1, 1>>>(non_init_ptr);
		cudaDeviceSynchronize();

	}


  // Multiply the arrays A and B on GPU and save the result in C
  // C(m,n) = A(m,k) * B(k,n)




  static void MatrixMul(bool transA, bool transB, const float *A, const float *B, float *C,
		  const int m, const int k, const int n, const float* scalarA = nullptr, const float* scalarB = nullptr, int lda = 0, int ldb = 0, int ldc = 0) {
	  if (lda == 0 ) lda = m;
	  if (ldb == 0 ) ldb = n;
	  if (ldc == 0 ) ldc = m;

     float alf = 1;
     float bet = 1;
     float *alpha = scalarA  ? const_cast<float*>(scalarA) : nullptr;  //assign the scalar
     float *beta =  scalarB  ? const_cast<float*>(scalarB) : nullptr;  //same
     // Create a handle for CUBLAS
     cublasHandle_t handle;
     cublasCreate(&handle);
     bool freeA = false;
     bool freeB = false;

     if (alpha || beta) { 			//if either of them were assigned set pointer_mode_to device
    	 cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    	 //if one of them are NOT assigned (create on gpu == 1)

			if (!alpha) {
				std::cout << " sorry you need to use scalars on bothsides of mul expression idk why this thing is crap"  << std::endl;
				createScalarOne(alpha);
				freeA = true;
			}
			else if (!beta) {
				std::cout << " sorry you need to use scalars on bothsides of mul expression idk why this thing is crap"  << std::endl;

				createScalarOne(beta);
				freeB = true;
			}
     }
     else {
    	 cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    	 //else assign to the 1 variable on the stack
    	 alpha = &alf;
    	 beta = &bet;
     }
     //convert bool to cuda enum
     auto TRANS_A =  transA ? CUBLAS_OP_T : CUBLAS_OP_N;
     auto TRANS_B =  transA ? CUBLAS_OP_T : CUBLAS_OP_N;

     // Do the actual multiplication
     cublasSgemm(handle, TRANS_A, TRANS_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
     cudaDeviceSynchronize();
     cublasDestroy(handle);

     //if we generated a scalar for the computation delete it
     if (freeA) cudaFree(const_cast<float*>(alpha));
     if (freeB) cudaFree(const_cast<float*>(beta));
}



	template<class ranks>
	static int calc_size(ranks R, int order) {
		if (order == 0) {
			return 0;
		}

		int sz = 1;
		for (int i = 0; i < order; ++i) {
			sz *= R[i];
		}
		return sz;
	}

	template<class T, class RANKS>
	static void print(const T* ary, const RANKS ranks, int order, int print_length) {
		int sz = calc_size(ranks, order);
			T* print = new T[sz];

			cudaMemcpy(print, ary, sizeof(T) * sz, cudaMemcpyDeviceToHost);

		BC::print(print, ranks, order, print_length);
		delete[] print;
	}
 };
}
#endif //BLACKCAT_GPU_ENABLED

#endif /* MATHEMATICS_CPU_H_ */
