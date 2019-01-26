/*
 * Device.h
 *
 *  Created on: Jan 24, 2019
 *      Author: joseph
 */

#ifdef __CUDACC__
#ifndef BC_CONTEXT_DEVICE_H_
#define BC_CONTEXT_DEVICE_H_

#include <cuda.h>
#include <cuda_runtime.h>

namespace BC {
namespace context {

template<class Allocator>
class Device : Allocator {

    cublasHandle_t m_cublas_handle = 0;
    cudaStream_t   m_stream = 0;

public:

    const cublasHandle_t& get_blas_handle() const {
    	return m_cublas_handle;
    }

    cublasHandle_t& get_blas_handle() {
    	return m_cublas_handle;
    }

    const cudaStream_t& get_stream() const {
    	return m_stream;
    }
    cudaStream_t& get_stream() {
    	return m_stream;
    }

    const Allocator& get_allocator() const {
    	return static_cast<const Allocator&>(*this);
    }

    Allocator& get_allocator() {
    	return static_cast<Allocator&>(*this);
    }


    Device(const Allocator& alloc_) : m_allocator(alloc_) {
        cublasCreate(&m_cublas_handle);
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    }
    Device(const Allocator& alloc_) : m_allocator(alloc_) {
        cublasCreate(&m_cublas_handle);
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    }
    Device(Allocator&& alloc_) : m_allocator(std::move(alloc_)) {
        cublasCreate(&m_cublas_handle);
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    }
    Device(Allocator&& alloc_) : m_allocator(std::move(alloc_)) {
        cublasCreate(&m_cublas_handle);
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    }

    ~Device() {
        cublasDestroy(handle);
    }

};


}
}



#endif /* DEVICE_H_ */
#endif
