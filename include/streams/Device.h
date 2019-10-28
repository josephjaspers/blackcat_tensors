/*
 * Device.h
 *
 *  Created on: Jan 24, 2019
 *	  Author: joseph
 */

#ifdef __CUDACC__
#ifndef BC_CONTEXT_DEVICE_H_
#define BC_CONTEXT_DEVICE_H_

#include "Host_Stream.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublas.h>

#include <memory>
#include <future>

namespace BC {
namespace streams {

template<class> class Stream;

template<>
class Stream<device_tag> {

	struct Device_Stream_Contents {

		HostStream	   m_host_stream;
		cublasHandle_t m_cublas_handle;
		cudaStream_t   m_stream_handle=nullptr;
		cudaEvent_t	m_event_handle=nullptr;

		BC::allocators::Stack_Allocator<device_tag> m_workspace;

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
	using allocator_type = BC::allocators::Stack_Allocator<device_tag>;

	BC::allocators::Stack_Allocator<device_tag>& get_allocator() {
		return device_contents.get()->m_workspace;
	}

	template<class RebindType>
	auto get_allocator_rebound() {
		return typename allocator_type::template rebind<RebindType>::other(get_allocator());
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


   operator cudaStream_t() const {
		return device_contents.get()->m_stream_handle;
	}


	void set_stream(Stream dev) {
		device_contents = dev.device_contents;
	}

	void record_event() {
		BC_CUDA_ASSERT(cudaEventRecord(
				device_contents.get()->m_event_handle,
				device_contents.get()->m_stream_handle));
	}

	void wait_event(Stream& stream) {
		BC_CUDA_ASSERT(cudaStreamWaitEvent(
				device_contents.get()->m_stream_handle,
				stream.device_contents.get()->m_event_handle, 0));
	}

	void wait_stream(Stream& stream) {
		stream.record_event();
		BC_CUDA_ASSERT(cudaStreamWaitEvent(
				device_contents.get()->m_stream_handle,
				stream.device_contents.get()->m_event_handle, 0));
	}

	void wait_event(cudaEvent_t event) {
		BC_CUDA_ASSERT(cudaStreamWaitEvent(
				device_contents.get()->m_stream_handle,
				event, 0));
	}

	bool is_default() {
		return device_contents.get()->m_stream_handle == 0;
	}

	void create() {
		device_contents = std::shared_ptr<Device_Stream_Contents>(
				new Device_Stream_Contents(true));
	}
	void destroy() {
		//'reset' to default
		device_contents = get_default_contents();
	}

	void sync() {
		if (!is_default())
			BC_CUDA_ASSERT(cudaStreamSynchronize(device_contents.get()->m_stream_handle));
	}

	//Functions (even functions using the internal stream)
	//should use 'enqueue'
	template<class Function>
	void enqueue(Function f) {
		f();
	}

	template<
		class function,
		class=std::enable_if_t<std::is_void<decltype(std::declval<function>()())>::value>>
	void enqueue_callback(function func) {

		if (is_default()){
			func();
			return;
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
	template<
		class function,
		class=std::enable_if_t<!std::is_void<decltype(std::declval<function>()())>::value>, int ADL=0>
	auto enqueue_callback(function func) {
		std::promise<decltype(func())> promise;

		if (is_default()){
			promise.set_value(func());
			return promise.get_future();
		}


		auto future = promise.get_future();
		BC_CUDA_ASSERT(cudaEventRecord(
				device_contents.get()->m_event_handle,
				device_contents.get()->m_stream_handle));

		device_contents.get()->m_host_stream.push(
				BC::traits::bind(
				[this, func](std::promise<decltype(func())> promise) {
					cudaEventSynchronize(this->device_contents.get()->m_event_handle);
					promise.set_value(func());
		}, std::move(promise))
		);
		return future;
	}

	bool operator == (const Stream& dev) {
		return device_contents == dev.device_contents;
	}

	bool operator != (const Stream& dev) {
		return device_contents != dev.device_contents;
	}
};


}
}


#endif /* DEVICE_H_ */
#endif
