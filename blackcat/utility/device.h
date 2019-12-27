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


namespace bc {
namespace utility {

template<class SystemTag>
struct Utility;

template<>
struct Utility<device_tag> {

	template<class T>
	static void HostToDevice(T* t, const T* u, bc::size_t  size = 1) {
		cudaMemcpy(t, u, sizeof(T) * size, cudaMemcpyHostToDevice);
	}

	template<class T>
	static void DeviceToHost(T* t, const T* u, bc::size_t  size = 1) {
		cudaMemcpy(t, u, sizeof(T) * size, cudaMemcpyDeviceToHost);
	}
};


}
}




#endif /* DEVICE_H_ */
#endif
