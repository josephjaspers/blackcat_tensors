/*
 * Device.h
 *
 *  Created on: Jan 24, 2019
 *      Author: joseph
 */

#ifdef __CUDACC__
#ifndef BC_CONTEXT_DEVICE_H_
#define BC_CONTEXT_DEVICE_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <vector>

#include "Context_Impl.cu"

namespace BC {
namespace context {
namespace device_globals {

struct Alpha_Beta_Pair_Recycler {
	std::vector<float*> recycler;
	std::mutex locker;

	float* allocate() {

		float* data_ptr;
		if (recycler.empty()) {
	        cudaMallocManaged((void**) &data_ptr, sizeof(float));
		} else {
			locker.lock();
			data_ptr = recycler.back();
			recycler.pop_back();
			locker.unlock();
		}
		return data_ptr;
	}
	void deallocate(float* data_ptr) {
		locker.lock();
		recycler.push_back(data_ptr);
		locker.unlock();
	}
} alpha_beta_pair_recycler;

}

struct Device_Stream_Contents {
	 cublasHandle_t m_cublas_handle;
	 cudaStream_t   m_stream_handle;
	 float*         m_scalar_buffer=nullptr;

	 Device_Stream_Contents(bool init_stream=false, bool init_scalars=true) {
		 cublasCreate(&m_cublas_handle);
		 if (init_stream) {
			 cudaStreamCreate(&m_stream_handle);
		 }
		 if (init_scalars) {
			 m_scalar_buffer = device_globals::alpha_beta_pair_recycler.allocate();
		 }
	 }

	 template<class T>
	 T* get_scalar_buffer() {
		 static_assert(sizeof(T)<=sizeof(float), "MAXIMUM OF 32 BITS");
		 return reinterpret_cast<T*>(m_scalar_buffer);
	 }

	 ~Device_Stream_Contents() {
		 cublasDestroy(m_cublas_handle);
		 cudaStreamDestroy(m_stream_handle);

		 //remember we only need to deallocate alpha
		 device_globals::alpha_beta_pair_recycler.deallocate(m_scalar_buffer);
	 }
};

namespace device_globals {
	std::shared_ptr<Device_Stream_Contents> default_contents =
			std::shared_ptr<Device_Stream_Contents>(new Device_Stream_Contents());
}


struct  Device {

	template<class scalar_t, int value>
	static const scalar_t* scalar_constant() {
		static scalar_t* scalar_constant_ = nullptr;

		if (!scalar_constant_) {
			std::mutex locker;
			locker.lock();
			if (!scalar_constant_){
				scalar_t tmp_val = value;
				cudaMallocManaged((void**) &scalar_constant_, sizeof(scalar_t));
				cudaMemcpy(scalar_constant_, &tmp_val, sizeof(scalar_t), cudaMemcpyHostToDevice);
			}
			locker.unlock();
		}
		return scalar_constant_;
	}


private:
	std::shared_ptr<Device_Stream_Contents> device_contents =
			device_globals::default_contents;
public:

	template<class T>
	T* scalar_alpha(T alpha_value) {
		T* buffer = device_contents.get()->get_scalar_buffer<T>();
		set_scalar_value(buffer, alpha_value, device_contents.get()->m_stream_handle);
		return buffer;
	}

    const cublasHandle_t& get_cublas_handle() const {
    	return device_contents.get()->m_cublas_handle;
    }

    cublasHandle_t& get_cublas_handle() {
    	return device_contents.get()->m_cublas_handle;
    }

    const cudaStream_t& get_stream() const {
    	return device_contents.get()->m_stream_handle;
    }
    cudaStream_t& get_stream() {
    	return device_contents.get()->m_stream_handle;
    }

    void set_stream(Device& dev) {
    	device_contents = dev.device_contents;
    }

    bool is_default_stream() {
    	return device_contents.get()->m_stream_handle == 0;
    }


    void create_stream() {
    	device_contents = std::shared_ptr<Device_Stream_Contents>(
    			new Device_Stream_Contents(true));
    }
    void destroy_stream() {
    	//'reset' to default
    	device_contents = device_globals::default_contents;
    }

    void sync_stream() {
    	if (!is_default_stream())
    		cudaStreamSynchronize(device_contents.get()->m_stream_handle);
    }

    bool operator == (const Device& dev) {
    	return device_contents == dev.device_contents;
    }

    bool operator != (const Device& dev) {
		return device_contents != dev.device_contents;
	}

    Device() = default;
    Device(const Device& dev) = default;
    Device(Device&&) = default;
};


}
}


#endif /* DEVICE_H_ */
#endif
