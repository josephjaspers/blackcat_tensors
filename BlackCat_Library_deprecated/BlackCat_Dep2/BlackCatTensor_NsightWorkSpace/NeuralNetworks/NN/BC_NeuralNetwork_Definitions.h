#ifndef BLACKCAT_NeuralNetwork_definitions
#define BLACKCAT_NeuralNetwork_definitions

//DO NOT MODIFY
enum Layer_Implementations { FF, RU, GRU, LSTM, CNN };
//enum nonLinearity { sigmoid, tan_h, relu };
//Utilized for saving/writing NN's

#include <vector>


//#define CUDA_GPU_ENABLED
//--Currently not working --- in development
#ifdef CUDA_GPU_ENABLED				//Enables Tensor Math to be run on GPU
typedef Tensor<float, GPU> tensor;
typedef Matrix<float, GPU> mat;
typedef Vector<float, GPU> vec;
typedef std::vector<Tensor<unsigned>> max_ids;

typedef float scalar;

#else
typedef Tensor_Queen<double> tensor;
typedef Matrix<double> mat;
typedef Vector<double> vec;
typedef double scalar;
#endif


#endif
