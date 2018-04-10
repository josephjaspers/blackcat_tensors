#ifndef BLACKCAT_NeuralNetwork_definitions
#define BLACKCAT_NeuralNetwork_definitions

#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

//DO NOT MODIFY
enum Layer_Implementations { FF, RU, GRU, LSTM, CNN };
enum nonLinearity { sigmoid, tan_h, relu };
//Utilized for saving/writing NN's

#include <vector>

//#define GPU_Tensors
#ifdef CUDA_GPU_ENABLED				//Enables Tensor Math to be run on GPU
typedef Tensor<float> tensor;
typedef Matrix<float> mat;
typedef Vector<float> vec;
typedef Scalar<float> scalar;


#else
typedef Tensor<double> tensor;
typedef Matrix<double> mat;
typedef Vector<double> vec;
typedef Scalar<double> scalar;

#endif


#endif
