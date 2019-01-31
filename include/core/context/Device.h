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
#include <memory>
#include <mutex>

namespace BC {
namespace context {
namespace device_globals {

template<class value_type, int id> //buffer name should be either 'A' (for alpha) or 'B' (for beta)
static value_type* constant_buffer() {

	struct cuda_destroyer {
		void operator () (value_type* val) {
			cudaFree(val);
		}
	};

	static std::unique_ptr<value_type, cuda_destroyer> buf;

	if (!buf) {
		std::mutex locker;
		locker.lock();
		if (!buf) { //second if statement is intentional
			cudaMalloc((void**) &buf.get(), sizeof(value_type));
		}
		locker.unlock();
	}
	return buf.get();
}


cublasHandle_t DEFAULT_CUBLAS_HANDLE;
//lapackHandle_t DEFAULT_LAPACK_HANDLE  //uncomment once we begin to support Lapack
cudaStream_t   DEFAULT_STREAM;


struct Default_Device_Context_Parameters {

	std::shared_ptr<cublasHandle_t> cublas_handle;
	std::shared_ptr<cudaStream_t>   stream_handle;

	Default_Device_Context_Parameters() {

		cublasCreate(&DEFAULT_CUBLAS_HANDLE);
		cublas_handle = std::shared_ptr<cublasHandle_t>(&DEFAULT_CUBLAS_HANDLE, [](cublasHandle_t*) {});
	}
	~Default_Device_Context_Parameters() {
		cublasDestroy(DEFAULT_CUBLAS_HANDLE);
	}

} default_context_parameters;


} //end of namespace 'device globals'

template<class Allocator>
struct  Device : public Allocator  {

	using value_type = typename Allocator::value_type;

private:
	std::shared_ptr<cublasHandle_t> m_cublas_handle = device_globals::default_context_parameters.cublas_handle;
	std::shared_ptr<cudaStream_t> m_stream 	        = device_globals::default_context_parameters.stream_handle;

    //The temporary scalars used in BLAS calls,
    //we allocate before hand as they cudaMalloc calls cudaDeviceSynchronize (which kills performance)
    value_type*    alpha_buffer   = device_globals::constant_buffer<value_type, 0>();
    value_type*    beta_buffer    = device_globals::constant_buffer<value_type, 1>();

public:

    const auto& get_blas_handle() const {
    	return m_cublas_handle;
    }

    auto& get_blas_handle() {
    	return m_cublas_handle;
    }

    const auto& get_stream() const {
    	return m_stream;
    }
    auto& get_stream() {
    	return m_stream;
    }

    const Allocator& get_allocator() const {
    	return static_cast<const Allocator&>(*this);
    }

    Allocator& get_allocator() {
    	return static_cast<Allocator&>(*this);
    }

    Device() = default;
    Device(const Device& dev)
     : Allocator(dev.m_allocator),
       m_cublas_handle(dev.m_cublas_handle),
       m_stream(dev.m_stream),
       alpha_buffer(dev.alpha_buffer),
       beta_buffer(dev.beta_buffer) {}

    Device(const Allocator& alloc_) : Allocator(alloc_) {}
};


}
}



#endif /* DEVICE_H_ */
#endif
