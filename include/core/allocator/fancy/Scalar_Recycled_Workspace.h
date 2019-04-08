/*
 * Scalar_Recycled_Workspace.h
 *
 *  Created on: Apr 6, 2019
 *      Author: joseph
 */

#ifndef SCALAR_RECYCLED_WORKSPACE_H_
#define SCALAR_RECYCLED_WORKSPACE_H_

#include "Workspace.h"

namespace BC {
namespace allocator {
namespace fancy {

#ifdef __CUDACC__
namespace device_globals {
struct Scalar_Recycler {

	template<class T>
	static std::vector<T*>& get_recycler() {
		static std::vector<T*> recycler_instance;
		return recycler_instance;
	}

	template<class T>
	static T* allocate() {
		static std::mutex locker;
		T* data_ptr = nullptr;

		if (get_recycler<T>().empty()) {
			BC_CUDA_ASSERT(cudaMalloc((void**) &data_ptr, sizeof(T)));
		} else {
			locker.lock();
			data_ptr = get_recycler<T>().back();
			get_recycler<T>().pop_back();
			locker.unlock();
		}
		return data_ptr;
	}

	template<class T>
	static void deallocate(T* data_ptr) {
		static std::mutex locker;
		locker.lock();
		get_recycler<T>().push_back(data_ptr);
		locker.unlock();
	}
};
}
#endif

#ifndef BC_DEVICE_BUFFER_SIZE
#define BC_DEVICE_BUFFER_SIZE 32
#endif

#ifndef BC_HOST_BUFFER_SIZE
#define BC_HOST_BUFFER_SIZE 128
#endif


template<class SystemTag>
class Scalar_Recycled_Workspace;

template<>
class Scalar_Recycled_Workspace<host_tag> : public Workspace<host_tag> {

	using buffer_type = Byte[BC_HOST_BUFFER_SIZE];
	buffer_type buffer;

public:

	template<class T>
	T* get_alpha_buffer() {
		static_assert(sizeof(T) <= sizeof(buffer_type), "MAX SCALAR SIZE == 32");
		return reinterpret_cast<T*>(buffer);
	}
};

#ifdef  __CUDACC__
template<>
class Scalar_Recycled_Workspace<device_tag> : public Workspace<device_tag> {

	using buffer_type = Byte[BC_DEVICE_BUFFER_SIZE];
	buffer_type* buffer = device_globals::Scalar_Recycler::template allocate<buffer_type>();

public:

	template<class T>
	T* get_alpha_buffer() {
		static_assert(sizeof(T) <= sizeof(buffer_type), "MAX SCALAR SIZE == 32");
		return reinterpret_cast<T*>(buffer);
	}

	~Scalar_Recycled_Workspace<device_tag>() {
		device_globals::Scalar_Recycler::deallocate<buffer_type>(buffer);
	}
};
#endif

}
}
}




#endif /* SCALAR_RECYCLED_WORKSPACE_H_ */
