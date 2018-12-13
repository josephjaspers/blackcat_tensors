/*
 * Host.h
 *
 *  Created on: Dec 12, 2018
 *      Author: joseph
 */

#ifndef BC_UTILITY_HOST_H_
#define BC_UTILITY_HOST_H_

namespace BC {
namespace utility {

struct Host {
   template<typename T>
	static T stack_allocate(T t) {
		return t;
	}

    template<typename T>
    static void deallocate(T t) {
        //empty
    }

	template<class T, class U, class V>
	static void copy(T* to, U* from, V size) {
		__BC_omp_for__
		for (int i = 0; i < size; ++i) {
			to[i] = from[i];
		}
		__BC_omp_bar__
	}

    template<class T, class U>
    static void HostToDevice(T* device_ptr, U* host_ptr, int size=1) {
        copy(device_ptr, host_ptr, size);
    }
    template<class T, class U>
    static void DeviceToHost(T* host_ptr, U* device_ptr, int size=1) {
        copy(host_ptr, device_ptr, size);
    }
    template<class T>
    static T extract(T* data_ptr, int index) {
    	return data_ptr[index];
    }
};

}
}



#endif /* HOST_H_ */
