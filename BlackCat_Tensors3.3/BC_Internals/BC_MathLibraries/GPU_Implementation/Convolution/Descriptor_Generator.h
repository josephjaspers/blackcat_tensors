/*
 * Descriptor_Generator.h
 *
 *  Created on: Jun 25, 2018
 *      Author: joseph
 */

#ifndef DESCRIPTOR_GENERATOR_H_
#define DESCRIPTOR_GENERATOR_H_

namespace BC{
namespace NN {


template<class Tensor>
std::enable_if_t<Tensor::DIMS() == 4, cudnnTensorDescriptor_t> descriptor_batched(const Tensor& tensor) {
	  cudnnHandle_t cudnn;
	  cudnnCreate(&cudnn);

	  cudnnTensorDescriptor_t descriptor;
	  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
	  checkCUDNN(cudnnSetTensor3dDescriptor(input_descriptor,
	                                        /*format=*/CUDNN_TENSOR_NHWC,
	                                        /*dataType=*/CUDNN_DATA_FLOAT,
	                                        /*batch_size=*/tensor.outer_dimension(),
	                                        /*channels=*/tensor.dimension(2),
	                                        /*image_height=*/tensor.rows,
	                                        /*image_width=*/tensor.cols));
}
template<class Tensor>
std::enable_if_t<Tensor::DIMS() == 4, cudnnTensorDescriptor_t> descriptor_batched(const Tensor& tensor) {
	  cudnnHandle_t cudnn;
	  cudnnCreate(&cudnn);

	  cudnnTensorDescriptor_t descriptor;
	  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
	  checkCUDNN(cudnnSetTensor3dDescriptor(input_descriptor,
	                                        /*format=*/CUDNN_TENSOR_NHWC,
	                                        /*dataType=*/CUDNN_DATA_FLOAT,
	                                        /*batch_size=*/tensor.outer_dimension(),
	                                        /*channels=*/tensor.dimension(2),
	                                        /*image_height=*/tensor.rows,
	                                        /*image_width=*/tensor.cols));
}
template<class Tensor>
std::enable_if_t<Tensor::DIMS() == 4, cudnnTensorDescriptor_t> descriptor_batched(const Tensor& tensor) {
	  cudnnHandle_t cudnn;
	  cudnnCreate(&cudnn);

	  cudnnTensorDescriptor_t descriptor;
	  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
	  checkCUDNN(cudnnSetTensor3dDescriptor(input_descriptor,
	                                        /*format=*/CUDNN_TENSOR_NHWC,
	                                        /*dataType=*/CUDNN_DATA_FLOAT,
	                                        /*batch_size=*/tensor.outer_dimension(),
	                                        /*channels=*/tensor.dimension(2),
	                                        /*image_height=*/tensor.rows,
	                                        /*image_width=*/tensor.cols));
}
template<class Tensor>
std::enable_if_t<Tensor::DIMS() == 4, cudnnTensorDescriptor_t> descriptor_batched(const Tensor& tensor) {
	  cudnnHandle_t cudnn;
	  cudnnCreate(&cudnn);

	  cudnnTensorDescriptor_t descriptor;
	  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
	  checkCUDNN(cudnnSetTensor3dDescriptor(input_descriptor,
	                                        /*format=*/CUDNN_TENSOR_NHWC,
	                                        /*dataType=*/CUDNN_DATA_FLOAT,
	                                        /*batch_size=*/tensor.outer_dimension(),
	                                        /*channels=*/tensor.dimension(2),
	                                        /*image_height=*/tensor.rows,
	                                        /*image_width=*/tensor.cols));
}
}
}



#endif /* DESCRIPTOR_GENERATOR_H_ */
