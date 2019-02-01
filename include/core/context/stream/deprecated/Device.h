/*
 * Device.h
 *
 *  Created on: Jan 13, 2019
 *      Author: joseph
 */

#ifdef __CUDACC__
#ifndef BC_STREAMS_DEVICE_H_
#define BC_STREAMS_DEVICE_H_


#include <cuda_runtime_api.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>


namespace BC {
namespace streams {

/*
 * Wrapper for CudaStream
 */

struct DeviceQueue : QueueInterface<DeviceQueue> {

	cudaStream_t m_stream;





	bool active() const {
		return bool(m_stream);
	}

	void synchronize() {
		cudaStreamSynchronize(m_stream);
	}

	void init() {
		if (m_stream) {
			cudaStreamDestroy(m_stream);
		}
		cudaStreamCreate(&m_stream);
	}


	~DeviceQueue() {
		if (m_stream) {
			cudaStreamDestroy(m_stream);
		}
	}


};


}
}



#endif /* DEVICE_H_ */
#endif
