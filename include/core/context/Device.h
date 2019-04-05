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
#include <vector>

#include "HostQueue.h"
#include "Context_Impl.cu"

namespace BC {
namespace context {
namespace device_globals {

struct Scalar_Recycler {
	static std::vector<float*>& get_recycler() {
		static std::vector<float*> recycler_instance;
		return recycler_instance;
	}
	static std::mutex& get_locker() {
		static std::mutex locker_instance;
		return locker_instance;
	}
	static float* allocate() {

		float* data_ptr;
		if (get_recycler().empty()) {
			BC_CUDA_ASSERT(cudaMallocManaged((void**) &data_ptr, sizeof(float)));
		} else {
			get_locker().lock();
			data_ptr = get_recycler().back();
			get_recycler().pop_back();
			get_locker().unlock();
		}
		return data_ptr;
	}
	static void deallocate(float* data_ptr) {
		get_locker().lock();
		get_recycler().push_back(data_ptr);
		get_locker().unlock();
	}
};

}

class  Device {

	struct Device_Stream_Contents {
		using Byte = BC::context::Byte;

		HostQueue	   m_host_stream;
		cublasHandle_t m_cublas_handle;
		cudaStream_t   m_stream_handle=nullptr;
		cudaEvent_t    m_event_handle		  =nullptr;
		float*         m_scalar_buffer=nullptr;
		Workspace<device_tag> m_workspace;

		Polymorphic_Allocator<Byte, device_tag> m_allocator;

		Device_Stream_Contents(bool init_stream=true, bool init_scalars=true) {
			cublasCreate(&m_cublas_handle);
			BC_CUDA_ASSERT((cublasSetPointerMode(m_cublas_handle, CUBLAS_POINTER_MODE_DEVICE)));

			 if (init_stream) {
				 m_host_stream.init();
				 BC_CUDA_ASSERT(cudaStreamCreate(&m_stream_handle));
				 cublasSetStream(m_cublas_handle, m_stream_handle);
			 }
			 if (init_scalars) {
				 m_scalar_buffer = device_globals::Scalar_Recycler::allocate();
			 }
		 }

		 template<class T>
		 T* get_scalar_buffer() {
			 static_assert(sizeof(T)<=sizeof(float), "MAXIMUM OF 32 BITS");
			 return reinterpret_cast<T*>(m_scalar_buffer);
		 }

		 ~Device_Stream_Contents() {
			 BC_CUDA_ASSERT(cublasDestroy(m_cublas_handle));
			 device_globals::Scalar_Recycler::deallocate(m_scalar_buffer);

			 if (m_stream_handle)
				 BC_CUDA_ASSERT(cudaStreamDestroy(m_stream_handle));

			 if (m_event_handle)
				 BC_CUDA_ASSERT(cudaEventDestroy(m_event_handle));
		 }
	};

	static std::shared_ptr<Device_Stream_Contents> get_default_contents() {
		thread_local std::shared_ptr<Device_Stream_Contents> default_contents =
				std::shared_ptr<Device_Stream_Contents>(new Device_Stream_Contents(false));
		return default_contents;
	}

	std::shared_ptr<Device_Stream_Contents> device_contents = get_default_contents();

public:

	using system_tag = device_tag;

	template<class T>
	T* scalar_alpha() {
		return device_contents.get()->get_scalar_buffer<T>();
	}

    Workspace<device_tag>& get_allocator() {
    	return device_contents.get()->m_workspace;
    }

    const cublasHandle_t& get_cublas_handle() const {
    	return device_contents.get()->m_cublas_handle;
    }

    cublasHandle_t& get_cublas_handle() {
    	return device_contents.get()->m_cublas_handle;
    }

    const cudaStream_t& get_stream() const {
    	return device_contents.get()->m_stream_handle;
    }
    cudaStream_t& get_stream() {
    	return device_contents.get()->m_stream_handle;
    }

    void set_stream(Device& dev) {
    	device_contents = dev.device_contents;
    }

    void stream_record_event() {
    	cudaEventRecord(
    			device_contents.get()->m_event_handle,
    			device_contents.get()->m_stream_handle
    			);
    }

    void stream_wait_event(Device& stream) {
    	cudaStreamWaitEvent(device_contents.get()->m_stream_handle,
    						stream.device_contents.get()->m_event_handle);
    }

    void stream_wait_stream(Device& stream) {
    	stream.stream_record_event();
    	cudaStreamWaitEvent(device_contents.get()->m_stream_handle,
    						stream.device_contents.get()->m_event_handle);
    }


    void stream_wait_event(cudaEvent_t event) {
    	cudaStreamWaitEvent(device_contents.get()->m_stream_handle, event);
    }


    bool is_default_stream() {
    	return device_contents.get()->m_stream_handle == 0;
    }


    void create_stream() {
    	device_contents = std::shared_ptr<Device_Stream_Contents>(
    			new Device_Stream_Contents(true));
    }
    void destroy_stream() {
    	//'reset' to default
    	device_contents = get_default_contents();
    }

    void sync_stream() {
    	if (!is_default_stream())
    		cudaStreamSynchronize(device_contents.get()->m_stream_handle);
    }

    template<class function>
    void push_job(function func) {

    	if (is_default_stream()){
    		func();
    	}

    	cudaEventRecord(device_contents.get()->m_event_handle, device_contents.get()->m_stream_handle);
    	device_contents.get()->m_host_stream.push(
    			[&, func]() {
    				cudaEventSynchronize(device_contents.get()->m_event_handle);
    				func();
    			}
    	);
    }

    bool operator == (const Device& dev) {
    	return device_contents == dev.device_contents;
    }

    bool operator != (const Device& dev) {
		return device_contents != dev.device_contents;
	}

    Device() = default;
    Device(const Device& dev) = default;
    Device(Device&&) = default;
};


}
}


#endif /* DEVICE_H_ */
#endif
