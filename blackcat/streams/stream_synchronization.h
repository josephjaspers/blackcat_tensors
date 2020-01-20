/*
 * Stream_Synchronization.h
 *
 *  Created on: Dec 16, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_STREAM_SYNCHRONIZATION_H_
#define BLACKCAT_STREAM_SYNCHRONIZATION_H_

#include "../common.h"

namespace bc {
namespace streams {

inline void host_sync() {
	BC_omp_bar__
}

inline void device_sync() {
	BC_IF_CUDA(cudaDeviceSynchronize();)
}

inline void synchronize() {
	host_sync();
	device_sync();
}

}
}




#endif /* STREAM_SYNCHRONIZATION_H_ */
