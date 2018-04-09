#include "MNIST_test.h"
#include "MNIST_test_MT.h"

#define BC_DISABLE_OPENMP

int main() {

	BC::MNIST_Test::percept_MNIST();
//	BC::MNIST_Test_MT::percept_MNIST();
	std::cout << "terminate success "<< std::endl;



}
