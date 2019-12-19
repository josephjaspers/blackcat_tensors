#ifdef __CUDACC__
#ifndef BC_CU_WISE_BENCH_H_
#define BC_CU_WISE_BENCH_H_
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda.h>


template<class T> __global__
void cuda_cwise_test_kernel(T* out, unsigned length, T* a, T* b, T* c, T* d) {

		//grid stride loop
	    auto i = blockIdx.x * blockDim.x + threadIdx.x;
	    for (; i < length; i += blockDim.x * gridDim.x) {
	        out[i] = a[i] + b[i] - c[i] / d[i];
	    }
}

namespace bc {
namespace benchmarks {


template<class scalar_t, class allocator>
auto cu_cwise(int size, int reps) {

	    using vec   = bc::Vector<scalar_t, allocator>;

	    vec a(size);
	    vec b(size);
	    vec c(size);
	    vec d(size);
	    vec e(size);

	    a.randomize(-1000, 1000);
	    b.randomize(-1000, 1000);
	    c.randomize(-1000, 1000);
	    d.randomize(-1000, 1000);
	    e.randomize(-1000, 1000);


		auto f = [&]() {

			cuda_cwise_test_kernel<scalar_t><<<bc::calculate_block_dim(size),bc::calculate_threads()>>>(
					a.data(), size, b.data(), c.data(), d.data(), e.data());

		};
		return timeit(f, reps);

}

}
}
#endif
#endif
