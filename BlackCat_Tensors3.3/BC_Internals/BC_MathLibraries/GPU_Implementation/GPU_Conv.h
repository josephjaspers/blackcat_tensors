/*
 * GPU_Conv.h
 *
 *  Created on: May 31, 2018
 *      Author: joseph
 */

#ifndef GPU_CONV_H_
#define GPU_CONV_H_

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

template<int dim> struct DESCRIPTORS;
template<> struct DESCRIPTORS<1> {
	using tensor = cudnnSetTensor1dDescriptor;
	using format = CUDNN_TENSOR_HWC;
};
template<> struct DESCRIPTORS<2> {
	using tensor = cudnnSetTensor2dDescriptor;
};
template<> struct DESCRIPTORS<3> {
	using tensor = cudnnSetTensor3dDescriptor;
};
template<> struct DESCRIPTORS<4> {
	using tensor = cudnnSetTensor4dDescriptor;
};
template<int dim> using tensor_descriptor = typename DESCRIPTORS<dim>::tensor;

template<int dimension>
void convolution() {

	//generate initial handle
	checkCUDNN(cudnnCreate(&cudnn));

	//input descriptor
	cudnnTensorDescriptor_t input_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
	checkCUDNN(tensor_descriptor<dimension>(input_descriptor,
	                                      /*format=*/CUDNN_TENSOR_HWC,
	                                      /*internalType=*/CUDNN_DATA_FLOAT,
	                                      /*batch_size=*/1,
	                                      /*channels=*/3,
	                                      /*image_height=*/image.rows,
	                                      /*image_width=*/image.cols));

	cudnnTensorDescriptor_t output_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
	checkCUDNN(tensor_descriptor<dimension>(output_descriptor,
	                                      /*format=*/CUDNN_TENSOR_NHWC,
	                                      /*internalType=*/CUDNN_DATA_FLOAT,
	                                      /*batch_size=*/1,
	                                      /*channels=*/3,
	                                      /*image_height=*/image.rows,
	                                      /*image_width=*/image.cols));

	cudnnFilterDescriptor_t kernel_descriptor;
	checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
	checkCUDNN(tensor_descriptor<dimension>(kernel_descriptor,
	                                      /*internalType=*/CUDNN_DATA_FLOAT,
	                                      /*format=*/CUDNN_TENSOR_NCHW,
	                                      /*out_channels=*/3,
	                                      /*in_channels=*/3,
	                                      /*kernel_height=*/3,
	                                      /*kernel_width=*/3));

	cudnnConvolutionDescriptor_t convolution_descriptor;
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
	checkCUDNN(tensor_descriptor<dimension>(convolution_descriptor,
	                                           /*pad_height=*/1,
	                                           /*pad_width=*/1,
	                                           /*vertical_stride=*/1,
	                                           /*horizontal_stride=*/1,
	                                           /*dilation_height=*/1,
	                                           /*dilation_width=*/1,
	                                           /*mode=*/CUDNN_CROSS_CORRELATION,
	                                           /*computeType=*/CUDNN_DATA_FLOAT));

	cudnnConvolutionFwdAlgo_t convolution_algorithm;
	checkCUDNN(
	    cudnnGetConvolutionForwardAlgorithm(cudnn,
	                                        input_descriptor,
	                                        kernel_descriptor,
	                                        convolution_descriptor,
	                                        output_descriptor,
	                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
	                                        /*memoryLimitInBytes=*/0,
	                                        &convolution_algorithm));

	const float alpha = 1, beta = 0;
	checkCUDNN(cudnnConvolutionForward(cudnn,
	                                   &alpha,
	                                   input_descriptor,
	                                   d_input,
	                                   kernel_descriptor,
	                                   d_kernel,
	                                   convolution_descriptor,
	                                   convolution_algorithm,
	                                   d_workspace,
	                                   workspace_bytes,
	                                   &beta,
	                                   output_descriptor,
	                                   d_output));

}



#endif /* GPU_CONV_H_ */
