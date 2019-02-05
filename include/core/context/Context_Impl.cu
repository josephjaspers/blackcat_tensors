
#ifndef BC_CONTEXT_IMPL_CU_
#define BC_CONTEXT_IMPL_CU_
#include <cuda.h>
#include <cuda_runtime.h>
namespace BC {

namespace gpu_impl {
__global__
void scalar_set_kernel(float* val, float val2){
	val[0] = val2;
}
}



void set_scalar_value(float* val, float val2, cudaStream_t stream) {
	gpu_impl::scalar_set_kernel<<<1,1, 0, stream>>>(val, val2);
}

}
#endif
