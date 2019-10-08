/*
 * Logging_Stream.h
 *
 *  Created on: Jul 26, 2019
 *	  Author: joseph
 */

#ifndef BLACKAT_TENSORS_STREAMS_LOGGING_STREAM_H_
#define BLACKAT_TENSORS_STREAMS_LOGGING_STREAM_H_

namespace BC {

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
struct Logging_Stream_Base {

};

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

	struct log_info {

		unsigned max_allocated; //maximum number of bytes allocated
		unsigned current_allocated;
	};

	template<class T>
	struct Allocator {

		using system_tag = SystemTag;
		using value_type = T;

		std::shared_ptr<log_info> info=
				std::shared_ptr<log_info>(new log_info {0, 0});

		template<class altT>
		struct rebind { using other = Allocator<T>; };

		template<class U>
		Allocator(const Allocator<U>& other): info(other.info) {}
		Allocator() = default;
		Allocator(const Allocator&) = default;
		Allocator(Allocator&&) = default;

		Allocator& operator = (const Allocator&) = default;
		Allocator& operator = (Allocator&&) = default;

		T* allocate(int size) {
			info->current_allocated += size * sizeof(value_type);
			if (info->current_allocated > info->max_allocated){
				info->max_allocated = info->current_allocated;
			}
			return nullptr;
		}

		void reserve(unsigned sz) {
			info->current_allocated += sz * sizeof(value_type);
			if (info->current_allocated > info->max_allocated){
				info->max_allocated = info->current_allocated;
			}
		}

		void deallocate(T* ptr, BC::size_t size) {
			BC_ASSERT(ptr == nullptr,
					"LOGGING_ALLOCATOR CAN ONLY DEALLOCATE NULLPTRS");
			BC_ASSERT(info->current_allocated >= size * sizeof(value_type),
					"BC_DEALLOCATION ERROR, DOUBLE DUPLICATION")
			info->current_allocated -= size * sizeof(value_type);
		}
	};


	Allocator<BC::allocators::Byte> allocator;

	Allocator<BC::allocators::Byte> get_allocator() const {
		return allocator;
	}

	template<class ValueType>
	Allocator<ValueType> get_allocator_rebound() const {
		return Allocator<ValueType>(allocator);
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
