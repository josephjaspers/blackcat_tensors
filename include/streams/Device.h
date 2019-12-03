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

		HostStream     m_host_stream;
		cublasHandle_t m_cublas_handle;
		cudaStream_t   m_stream_handle=nullptr;
		cudaEvent_t    m_event_handle=nullptr;

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

	using contents_handle_t = std::shared_ptr<Device_Stream_Contents>;

	static contents_handle_t get_default_contents() {
		thread_local contents_handle_t default_contents =
				contents_handle_t(new Device_Stream_Contents(false));
		return default_contents;
	}

	contents_handle_t m_contents = get_default_contents();

public:

	using system_tag = device_tag;
	using allocator_type = BC::allocators::Stack_Allocator<device_tag>;

	BC::allocators::Stack_Allocator<device_tag>& get_allocator() {
		return m_contents->m_workspace;
	}

	template<class RebindType>
	auto get_allocator_rebound() {
		return typename allocator_type::template rebind<RebindType>::other(get_allocator());
	}

	auto set_blas_pointer_mode_host() {
		cublasSetPointerMode(m_contents->m_cublas_handle, CUBLAS_POINTER_MODE_HOST);
	}
	auto set_blas_pointer_mode_device() {
		cublasSetPointerMode(m_contents->m_cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
	}

	cublasHandle_t get_cublas_handle() const {
		return m_contents->m_cublas_handle;
	}

	operator cudaStream_t() const {
		return m_contents->m_stream_handle;
	}

	void set_stream(Stream dev) {
		m_contents = dev.m_contents;
	}

	void record_event() {
		BC_CUDA_ASSERT(cudaEventRecord(
				m_contents->m_event_handle,
				m_contents->m_stream_handle));
	}

	void wait_event(Stream& stream) {
		BC_CUDA_ASSERT(cudaStreamWaitEvent(
				m_contents->m_stream_handle,
				stream.m_contents->m_event_handle, 0));
	}

	void wait_stream(Stream& stream) {
		stream.record_event();
		BC_CUDA_ASSERT(cudaStreamWaitEvent(
				m_contents->m_stream_handle,
				stream.m_contents->m_event_handle, 0));
	}

	void wait_event(cudaEvent_t event) {
		BC_CUDA_ASSERT(cudaStreamWaitEvent(
				m_contents->m_stream_handle,
				event, 0));
	}

	bool is_default() {
		return m_contents->m_stream_handle == 0;
	}

	void create() {
		m_contents =
				contents_handle_t(new Device_Stream_Contents(true));
	}

	void destroy() {
		m_contents = get_default_contents();
	}

	void sync()
	{
		if (!is_default()) {
			BC_CUDA_ASSERT(
					cudaStreamSynchronize(m_contents->m_stream_handle));
		}
	}

	template<class Function>
	void enqueue(Function f) {
		f();
	}

	template<
		class function,
		class=std::enable_if_t<
				std::is_void<
						decltype(std::declval<function>()())>::value>>
	void enqueue_callback(function func)
	{
		if (is_default()) {
			func();
			return;
		}

		BC_CUDA_ASSERT(cudaEventRecord(
				m_contents->m_event_handle,
				m_contents->m_stream_handle));

		m_contents->m_host_stream.push(
				[&, func]() {
					cudaEventSynchronize(m_contents->m_event_handle);
					func();
				}
		);
	}

	template<
		class function,
		class=std::enable_if_t<
				!std::is_void<
						decltype(std::declval<function>()())>::value>,
		int ADL=0>
	auto enqueue_callback(function func)
	{
		std::promise<decltype(func())> promise;

		if (is_default()){
			promise.set_value(func());
			return promise.get_future();
		}

		auto future = promise.get_future();
		BC_CUDA_ASSERT(cudaEventRecord(
				m_contents->m_event_handle,
				m_contents->m_stream_handle));

		m_contents->m_host_stream.push(
				BC::traits::bind(
				[this, func](std::promise<decltype(func())> promise) {
					cudaEventSynchronize(this->m_contents->m_event_handle);
					promise.set_value(func());
		}, std::move(promise)));

		return future;
	}

	bool operator == (const Stream& dev) {
		return m_contents == dev.m_contents;
	}

	bool operator != (const Stream& dev) {
		return m_contents != dev.m_contents;
	}
};


}
}


#endif /* DEVICE_H_ */
#endif
