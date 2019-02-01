/*
 * Device.h
 *
 *  Created on: Dec 12, 2018
 *      Author: joseph
 */
#ifdef __CUDACC__
#ifndef BC_UTILITY_DEVICE_H_
#define BC_UTILITY_DEVICE_H_

#include <memory>
#include <vector>
#include <mutex>

namespace BC {
namespace utility {
static std::vector<float*> scalar_recycler;

struct Device {



	static float* stack_allocate(float value) {

		static std::mutex locker;

		if (!scalar_recycler.empty()) {
			locker.lock();
			float* val = scalar_recycler.back();
			scalar_recycler.pop_back();
			locker.unlock();


			cudaDeviceSynchronize();
			if (*val != value)
				*val = value;

			return val;
		} else {

			float* t;
			cudaMallocManaged((void**) &t, sizeof(float));
			cudaMemcpy(t, &value, sizeof(float), cudaMemcpyHostToDevice);
			return t;
		}
    }

	static void deallocate(float* t) {
		static std::mutex locker;
		locker.lock();
		scalar_recycler.push_back(t);
		locker.unlock();

	}

	template<class T>
	static void HostToDevice(T* t, const T* u, BC::size_t  size = 1) {
		cudaMemcpy(t, u, sizeof(T) * size, cudaMemcpyHostToDevice);
	}
	template<class T>
	static void DeviceToHost(T* t, const T* u, BC::size_t  size = 1) {
		cudaMemcpy(t, u, sizeof(T) * size, cudaMemcpyDeviceToHost);
	}
	template<class T>
	static T extract(const T* data_ptr, BC::size_t  index) {
		T host_data;
		cudaMemcpy(&host_data, &data_ptr[index], sizeof(T), cudaMemcpyDeviceToHost);
		return host_data;
	}

};

}
}




#endif /* DEVICE_H_ */
#endif
