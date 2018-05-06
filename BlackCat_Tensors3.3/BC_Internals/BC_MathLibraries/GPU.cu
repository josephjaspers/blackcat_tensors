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


namespace BC {

class GPU :
	public GPU_Misc<GPU>,
	public GPU_Utility<GPU>,
	public GPU_BLAS<GPU> {
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

		struct n1 { template<class T, class F> static void copy(T to, const F from, int sz) {	gpu_impl::copy<<<blocks(sz),threads()>>>(to, from, sz); } };
		struct n2 { template<class T, class F> static void copy(T to, const F from, int sz) {	gpu_impl::copy2d<<<blocks(sz),threads()>>>(to, from); } };
		struct n3 { template<class T, class F> static void copy(T to, const F from, int sz) {	gpu_impl::copy3d<<<blocks(sz),threads()>>>(to, from); } };
		struct n4 { template<class T, class F> static void copy(T to, const F from, int sz) {	gpu_impl::copy4d<<<blocks(sz),threads()>>>(to, from); } };
		struct n5 { template<class T, class F> static void copy(T to, const F from, int sz) {	gpu_impl::copy5d<<<blocks(sz),threads()>>>(to, from); } };

		using run = std::conditional_t<(d <= 1), n1,
						std::conditional_t< d ==2, n2,
							std::conditional_t< d == 3, n3,
								std::conditional_t< d == 4, n4, n5>>>>;

		template<class T, class F>
		static void copy(T to, const F from) {
			run::copy(to,from, to.size());
			cudaDeviceSynchronize();
		}

		template<class T, template<class...> class F, class... set>
		static void copy(T to, F<set...> from) {
			run::copy(to,from, to.size());
			cudaDeviceSynchronize();
		}
		template<template<class...> class T, template<class...> class F, class... ts, class... fs>
		static void copy(T<ts...> to, F<fs...> from) {
			run::copy(to,from, to.size());
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
