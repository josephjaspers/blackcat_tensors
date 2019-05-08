/*
 * Device.h
 *
 *  Created on: Jan 24, 2019
 *      Author: joseph
 */

#ifdef __CUDACC__
#ifndef BC_CONTEXT_DEVICE_H_
#define BC_CONTEXT_DEVICE_H_

#include "HostStream.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublas.h>

#include <memory>

namespace BC {
namespace context {

class Device {

	struct Device_Stream_Contents {

		HostStream	   m_host_stream;
		cublasHandle_t m_cublas_handle;
		cudaStream_t   m_stream_handle =nullptr;
		cudaEvent_t    m_event_handle  =nullptr;

		BC::allocator::fancy::Workspace<device_tag> m_workspace;

		Device_Stream_Contents(bool init_stream=true) {
			BC_CUDA_ASSERT(cublasCreate(&m_cublas_handle));
			BC_CUDA_ASSERT(cudaEventCreate(&m_event_handle));
			BC_CUDA_ASSERT((cublasSetPointerMode(m_cublas_handle, CUBLAS_POINTER_MODE_DEVICE)));

			 if (init_stream) {
				 BC_CUDA_ASSERT(cudaStreamCreate(&m_stream_handle));
				 BC_CUDA_ASSERT(cublasSetStream(m_cublas_handle, m_stream_handle));
			 }
		 }

		 ~Device_Stream_Contents() {
			 BC_CUDA_ASSERT(cublasDestroy(m_cublas_handle));
			 BC_CUDA_ASSERT(cudaEventDestroy(m_event_handle));

			 if (m_stream_handle)
				 BC_CUDA_ASSERT(cudaStreamDestroy(m_stream_handle));
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
	using allocator_t = BC::allocator::fancy::Workspace<device_tag>;

	BC::allocator::fancy::Workspace<device_tag>& get_allocator() {
    	return device_contents.get()->m_workspace;
    }

    template<class RebindType>
    auto get_allocator_rebound() {
    	return typename allocator_t::template rebind<RebindType>::other(get_allocator());
    }

    auto set_blas_pointer_mode_host() {
    	cublasSetPointerMode(device_contents.get()->m_cublas_handle, CUBLAS_POINTER_MODE_HOST);
    }
    auto set_blas_pointer_mode_device() {
        cublasSetPointerMode(device_contents.get()->m_cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
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
    	BC_CUDA_ASSERT(cudaEventRecord(
    			device_contents.get()->m_event_handle,
    			device_contents.get()->m_stream_handle));
    }

    void stream_wait_event(Device& stream) {
    	BC_CUDA_ASSERT(cudaStreamWaitEvent(
    			device_contents.get()->m_stream_handle,
    			stream.device_contents.get()->m_event_handle, 0));
    }

    void stream_wait_stream(Device& stream) {
    	stream.stream_record_event();
    	BC_CUDA_ASSERT(cudaStreamWaitEvent(
    			device_contents.get()->m_stream_handle,
    			stream.device_contents.get()->m_event_handle, 0));
    }

    void stream_wait_event(cudaEvent_t event) {
    	BC_CUDA_ASSERT(cudaStreamWaitEvent(
    			device_contents.get()->m_stream_handle,
    			event, 0));
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
    		BC_CUDA_ASSERT(cudaStreamSynchronize(device_contents.get()->m_stream_handle));
    }

    template<class function>
    void stream_enqueue_callback(function func) {

    	if (is_default_stream()){
    		func();
    	}

    	BC_CUDA_ASSERT(cudaEventRecord(
    			device_contents.get()->m_event_handle,
    			device_contents.get()->m_stream_handle));

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
    Device & operator = (const Device& device) {
    	this->device_contents = device.device_contents;
    	return *this;
    }
    Device & operator = (Device&& device) {
    	this->device_contents = std::move(device.device_contents);
    	return *this;
    }
};


}
}


#endif /* DEVICE_H_ */
#endif
