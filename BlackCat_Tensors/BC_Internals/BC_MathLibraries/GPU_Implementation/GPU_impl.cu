
#ifdef __CUDACC__
#ifndef BC_GPU_IMPL
#define BC_GPU_IMPL

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

namespace BC {
namespace gpu_impl {

template<class T, class J> __global__
static void copy(T t, const J j, int sz) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (; i < sz; i += blockDim.x * gridDim.x) {
		t[i] = j[i];
	}
}



template<class T> __global__
static void eval(T t) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (; i < t.size(); i += blockDim.x * gridDim.x) {
		t[i];
	}
}

template<typename T> __global__  static void eval2d(T t) {
	int n = blockIdx.y * blockDim.y + threadIdx.y;
	for (; n < t.cols(); n += blockDim.y * gridDim.y) {

		int m = blockIdx.x * blockDim.x + threadIdx.x;
		for (; m < t.rows(); m += blockDim.x * gridDim.x) {
			t(m, n);
		}
	}
}
template<typename T> __global__ static void eval3d(T t) {
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	for (; k < t.dimension(2); k += blockDim.z * gridDim.z) {
		int n = blockIdx.y * blockDim.y + threadIdx.y;
		for (; n < t.cols(); n += blockDim.y * gridDim.y) {

			int m = blockIdx.x * blockDim.x + threadIdx.x;
			for (; m < t.rows(); m += blockDim.x * gridDim.x) {
				t(m,n,k);
			}
		}
	}
}
//dont know how to do this
template<typename T> __global__ static void eval4d(T t) {
	int l = blockIdx.z * blockDim.z + threadIdx.z;
	for (; l < t.dimension(3); l += blockDim.z * gridDim.z) {

		int k = blockIdx.y * blockDim.y + threadIdx.y;
		for (;k < t.dimension(2); k += blockDim.y * gridDim.y) {

			int n = blockIdx.x * blockDim.x + threadIdx.x;
			for (; n < t.cols(); n += blockDim.x * gridDim.x) {

				for (int m = 0; m < t.rows(); ++m) {
					t(m,n,k,l);
				}
			}
		}
	}
}
//don't know how to do this
template<typename T> __global__ static void eval5d(T t) {
	int p = blockIdx.z * blockDim.z + threadIdx.z;
	for (; p < t.dimension(4); p += blockDim.z * gridDim.z) {

		int l = blockIdx.y * blockDim.y + threadIdx.y;
		for (; l < t.dimension(3); l += blockDim.y * gridDim.y) {

			int k = blockIdx.x * blockDim.x + threadIdx.x;
			for (; k < t.dimension(2); k += blockDim.x * gridDim.x) {

				for (int n = 0; n < t.dimension(1); ++n) {

					for (int m = 0; m < t.dimension(0); ++m) {
						t(m, n, k, l, p);
					}
				}
			}
		}
	}
}

template<typename T, typename J> __global__
static void fill(T t, const J j) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (; i < t.size(); i += blockDim.x * gridDim.x) {
		t[i] = j;
	}
}

template<class T, class U, class V> __global__
static void scalar_mul(T* t, U* u, V* v) {
	*t = *u * *v;
}
template<class T, class U, class V> __global__
static void scalar_mul(T* t, U u, V* v) {
	*t = u * *v;
}
template<class T, class U, class V> __global__
static void scalar_mul(T* t, U* u, V v) {
	*t = *u * v;
}
template<class T, class U, class V> __global__
static void scalar_mul(T* t, U* u, V* v) {
	*t = u * v;
}



template<class T> __global__
static void randomize(T t, float lower_bound, float upper_bound, int seed) {

	 curandState_t state;
	  curand_init(seed, /* the seed controls the sequence of random values that are produced */
	              seed, /* the sequence number is only important with multiple cores */
	              1, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
	              &state);

	constexpr int floating_point_decimal_length = 10000;

	for (int i = 0; i < t.size(); ++i) {
		t[i] = curand(&state) % floating_point_decimal_length;
		t[i] /= floating_point_decimal_length;
		t[i] *= (upper_bound - lower_bound);
		t[i] += lower_bound;
	}
}

}


}

#endif
#endif //cudacc
