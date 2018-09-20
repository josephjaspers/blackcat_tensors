//#include "MNIST_test.h"
//#include "MNIST_Auto_Encoder.h"
#include "BlackCat_Tensors.h"
int main() {
//	BC::NN::MNIST_Test::percept_MNIST();
//	BC::NN::MNIST_Auto_Encoder_Test::percept_MNIST();

	BC::Scalar<float, BC::GPU> s;
	BC::Matrix<float, BC::GPU> a;
	BC::Matrix<float, BC::GPU> b;
	BC::Matrix<float, BC::GPU> c;


	a -= b * c;


	std::cout << "terminate success "<< std::endl;
}
