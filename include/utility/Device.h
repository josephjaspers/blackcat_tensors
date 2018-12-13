/*
 * Device.h
 *
 *  Created on: Dec 12, 2018
 *      Author: joseph
 */
#ifdef __CUDACC__
#ifndef BC_UTILITY_DEVICE_H_
#define BC_UTILITY_DEVICE_H_

namespace BC {
namespace utility {

struct Device {

    static float* stack_allocate(float value) {
        float* t;
        cudaMallocManaged((void**) &t, sizeof(float));
        cudaMemcpy(t, &value, sizeof(float), cudaMemcpyHostToDevice);
        return t;
    }

   template<typename T>
	static void deallocate(T* t) {
		cudaFree(t);
	}

	template<class T>
	static void HostToDevice(T* t, const T* u, int size = 1) {
		cudaDeviceSynchronize();
		cudaMemcpy(t, u, sizeof(T) * size, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
	}
	template<class T>
	static void DeviceToHost(T* t, const T* u, int size = 1) {
		cudaDeviceSynchronize();
		cudaMemcpy(t, u, sizeof(T) * size, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
	}
	template<class T>
	static T extract(const T* data_ptr, int index) {
		T host_data;
		cudaDeviceSynchronize();
		cudaMemcpy(&host_data, &data_ptr[index], sizeof(T), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		return host_data;
	}

};

}
}




#endif /* DEVICE_H_ */
#endif
