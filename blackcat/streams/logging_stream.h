/*
 * Logging_Stream.h
 *
 *  Created on: Jul 26, 2019
 *	  Author: joseph
 */

#ifndef BLACKAT_TENSORS_STREAMS_LOGGING_STREAM_H_
#define BLACKAT_TENSORS_STREAMS_LOGGING_STREAM_H_

namespace bc {

struct host_tag;
struct device_tag;

namespace streams {

/**
 * A Logging_Stream object does not actually allocate any memory.
 * It simply stores the required amount of memory that is has logged.
 * It is used in conjunction with a stream object when parsing an expression.
 *
 * Once an entire expression is created, it first attempts to evaluate the expression
 * while skipping all allocations and any of the enqueued functions.
 *
 * Once it has recorded all memory changes (allocations/deallocations) the logging allocator
 * has calculated the max amount of memory required for this calculation. We than use this
 * max_allocated in our Workspace Allocator (inside a stream) to reserve the amount required.
 *
 * This ensures that the maximum number of allocations required per expression is 1.
 * If the amount of memory reserved exceeds the required amount we do nothing.
 *
 */

template<class SystemTag>
struct Logging_Stream_Base {};

#ifdef __CUDACC__

template<>
struct Logging_Stream_Base<device_tag> {

	//This is required for matching the interface of Stream<device_tag>
	cublasHandle_t get_cublas_handle() const {
		return cublasHandle_t();
	}

	operator cudaStream_t() const {
		return cudaStream_t();
	}
};

#endif

template<class SystemTag>
struct Logging_Stream: Logging_Stream_Base<SystemTag> {

	using system_tag = SystemTag;
	using allocator_type = bc::allocators::Logging_Allocator<
			bc::allocators::Null_Allocator<
					bc::allocators::Byte,
					system_tag>>;

	allocator_type allocator;

	allocator_type get_allocator() const {
		return allocator;
	}

	template<class ValueType>
	auto get_allocator_rebound() const {
		return typename allocator_type::template
				rebind<ValueType>::other(allocator);
	}

	void set_blas_pointer_mode_device() const {};
	void set_blas_pointer_mode_host() const {};

	unsigned get_max_allocated() const {
		return allocator.info->max_allocated;
	}
	unsigned get_current_allocated() const {
		return allocator.info->current_allocated;
	}

	bool is_default() { return false; }
	void create() {}
	void destroy() {}
	void sync() {}

	template<class T>
	void set_stream(const T&) {}

	void record_event() {}

	template<class T>
	void wait_event(const T&) {}

	template<class T>
	void wait_stream(const T&) {}

	template<class Functor>
	void enqueue(const Functor& functor) {}

	template<class Functor>
	void enqueue_callback(const Functor& functor) {}

	bool operator == (const Logging_Stream<SystemTag>& dev) {
		return true;
	}
	bool operator != (const Logging_Stream<SystemTag>& dev) {
		return false;
	}
};

}  // namespace streams
}  // namespace BC

#endif /* LOGGING_STREAM_H_ */
