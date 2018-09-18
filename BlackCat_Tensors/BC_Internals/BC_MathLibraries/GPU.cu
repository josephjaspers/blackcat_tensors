#ifdef __CUDACC__
#ifndef MATHEMATICS_GPU_H_
#define MATHEMATICS_GPU_H_

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "Print.h"
#include "GPU_Implementation/GPU_impl.cu"
#include "GPU_Implementation/GPU_BLAS.h"
#include "GPU_Implementation/GPU_Misc.h"
#include "GPU_Implementation/GPU_Utility.h"
#include "GPU_Implementation/GPU_Constants.h"


#include "GPU_Implementation/GPU_Convolution.h"

namespace BC {

class GPU :
	public GPU_Misc<GPU>,
	public GPU_Utility<GPU>,
	public GPU_BLAS<GPU>,
	public GPU_Constants<GPU>,
	public GPU_Convolution<GPU> {
public:

	static constexpr int CUDA_BASE_THREADS = 256;

	static int blocks(int size) {
		return 1 + (int)(size / CUDA_BASE_THREADS);
	}
	static int threads(int sz = CUDA_BASE_THREADS) {
		return sz > CUDA_BASE_THREADS ? CUDA_BASE_THREADS : sz;
	}

	template<class T, class U>
	static void copy(T t, const U u, int sz) {
		gpu_impl::copy<<<blocks(sz),threads()>>>(t, u, sz);
		cudaDeviceSynchronize();
	}

	template<int d>
	struct dimension {

		struct n1 { template<class T> static void eval(T to) { gpu_impl::eval<<<blocks(to.size()),threads()>>>(to);   }};
		struct n2 { template<class T> static void eval(T to) { gpu_impl::eval2d<<<blocks(to.size()),threads()>>>(to); }};
		struct n3 { template<class T> static void eval(T to) { gpu_impl::eval3d<<<blocks(to.size()),threads()>>>(to); }};
		struct n4 { template<class T> static void eval(T to) { gpu_impl::eval4d<<<blocks(to.size()),threads()>>>(to); }};
		struct n5 { template<class T> static void eval(T to) { gpu_impl::eval5d<<<blocks(to.size()),threads()>>>(to); }};
		using run = std::conditional_t<(d <= 1), n1,
						std::conditional_t< d ==2, n2,
							std::conditional_t< d == 3, n3,
								std::conditional_t< d == 4, n4, n5>>>>;

		//These wonky specializations are essential for cuda to compile
		//Not sure why
		template<class T>
		static void eval(T to) {
			run::eval(to);
#ifndef __BC_PARALLEL_SECTION__
			cudaDeviceSynchronize();
#endif 
		}

		template<template<class...> class T, class... Ts>
		static void eval(T<Ts...> to) {
			run::eval(to);
			cudaDeviceSynchronize();
		}

	};

// THIS IS MANDATORY WITH CUDA COMPILATION FOR 9.1 --- THIS IS A BUG IN THE NVCC
	template<class T, template<class...> class U, class... set>
	static void copy(T t, U<set...> u, int sz) {
		gpu_impl::copy<<<blocks(sz),threads()>>>(t, u, sz);
		cudaDeviceSynchronize();
	}
	template<template<class...> class T, template<class...> class U, class... set, class... set1>
	static void copy(T<set1...> t, U<set...> u, int sz) {
		gpu_impl::copy<<<blocks(sz),threads()>>>(t, u, sz);
		cudaDeviceSynchronize();
	}



};

}

#endif /* MATHEMATICS_CPU_H_ */

#endif //if cudda cc defined
