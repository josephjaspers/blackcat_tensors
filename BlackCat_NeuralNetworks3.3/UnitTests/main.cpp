#include "MNIST_test.h"
#include "MNIST_test_MT.h"
#include "misspelledWords.h"

#define BC_DISABLE_OPENMP

int main() {

//	BC::MNIST_Test::percept_MNIST();
//	BC::MNIST_Test_MT::percept_MNIST();
	BC::Word_Test::test();
	std::cout << "terminate success "<< std::endl;

}
