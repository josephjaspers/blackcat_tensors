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

#ifndef NDEBUG
	#define BC_DEBUG_ASSERT(arg) if (!arg) throw std::invalid_argument(#arg " accessed failure ")
#else
	#define BC_DEBUG_ASSERT(arg)
#endif


struct  Device {

private:
	std::shared_ptr<cublasHandle_t> m_cublas_handle = device_globals::default_context_parameters.cublas_handle;
	std::shared_ptr<cudaStream_t> m_stream 	        = device_globals::default_context_parameters.stream_handle;

public:

	//Underscore represents 'dangerous method'

    const auto& get_blas_handle() const {
    	BC_DEBUG_ASSERT(m_cublas_handle.get());
    	return *(m_cublas_handle.get());
    }

    auto& get_blas_handle() {
    	BC_DEBUG_ASSERT(m_cublas_handle.get());
    	return *(m_cublas_handle.get());
    }

    const auto& get_stream() const {
    	BC_DEBUG_ASSERT(m_stream.get());
    	return *(m_stream.get());
    }
    auto& get_stream() {
    	BC_DEBUG_ASSERT(m_stream.get());
    	return *(m_stream.get());
    }

    bool is_default_stream() {
    	return bool(m_stream.get());
    }


    void create_stream() {
    	cudaStream_t* stream_ = nullptr;
    	cudaStreamCreate(stream_);

    	m_stream = std::shared_ptr<cudaStream_t>(
    			stream_,
    			[](cudaStream_t* del_stream_) { cudaStreamDestroy(*del_stream_); }
    	);

    	cublasHandle_t* handle_ = nullptr;
    	cublasCreate(handle_);
    	cublasSetStream(*handle_, *stream_);

    	m_cublas_handle = std::shared_ptr<cublasHandle_t>(
    			handle_,
    			[](cublasHandle_t* del_handle_){ cublasDestroy(*del_handle_); }
    	);



    }
    void delete_stream() {
    	m_stream.reset();
    	m_cublas_handle.reset();

    }

    void sync_stream() {
    	if (!is_default_stream())
    		cudaStreamSynchronize(*(this->m_stream.get()));
    }

    Device() = default;
    Device(const Device& dev)
     : m_cublas_handle(dev.m_cublas_handle),
       m_stream(dev.m_stream) {}
};


}
}


#undef BC_DEBUG_ASSERT
#endif /* DEVICE_H_ */
#endif
