#include "MNIST_test.h"
#include "MNIST_Auto_Encoder.h"
int main() {
	BC::NN::MNIST_Test::percept_MNIST();
//	BC::NN::MNIST_Auto_Encoder_Test::percept_MNIST();

	std::cout << "terminate success "<< std::endl;
}
