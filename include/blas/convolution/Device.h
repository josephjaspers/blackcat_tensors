/*
 * Device.h
 *
 *  Created on: Feb 26, 2019
 *      Author: joseph
 */

#ifndef BC_CORE_BLAS_CONVOLUTION_DEVICE_H_
#define BC_CORE_BLAS_CONVOLUTION_DEVICE_H_

#include <cudnn.h>

namespace BC {
namespace blas {
namespace convolution {

auto create_tensor_descriptor() {
	cudnnTensorDescriptor_t output_descriptor;
}
cudnnFilterDescriptor_t kernel_descriptor;
checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*out_channels=*/3,
                                      /*in_channels=*/3,
                                      /*kernel_height=*/3,
                                      /*kernel_width=*/3));

}
}
}



#endif /* DEVICE_H_ */
