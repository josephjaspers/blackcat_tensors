/*
 * Device_Scalar_Cacher.h
 *
 *  Created on: Mar 9, 2019
 *      Author: joseph
 */

#ifndef BC_CORE_ALLOCATOR_DEVICE_SCALAR_CACHER_H_
#define BC_CORE_ALLOCATOR_DEVICE_SCALAR_CACHER_H_

namespace BC {
namespace allocator {
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

	template<class T>
	static T* allocate() {
		static_assert(sizeof(T) <= sizeof(float),
				"MAXIMUM SIZE OF SCALAR_CACHING IS EQUAL TO THE SIZE OF FLOAT");

		float* data_ptr;
		if (get_recycler().empty()) {
			BC_CUDA_ASSERT(cudaMallocManaged((void**) &data_ptr, sizeof(float)));
		} else {
			get_locker().lock();
			data_ptr = get_recycler().back();
			get_recycler().pop_back();
			get_locker().unlock();
		}
		return reinterpret_cast<T*>(data_ptr);
	}

	template<class T>
	static void deallocate(T* data_ptr) {
		get_locker().lock();
		get_recycler().push_back(reinterpret_cast<float*>(data_ptr));
		get_locker().unlock();
	}
};

}
}
}



#endif /* DEVICE_SCALAR_CACHER_H_ */
