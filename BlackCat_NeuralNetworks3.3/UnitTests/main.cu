#include "MNIST_test.h"
//#include "MNIST_test_MT.h"
//#include "CharacterPrediction.h"
#include "ReccurrentFunctionMatcher.h"
#define BC_DISABLE_OPENMP

int main() {

	BC::NN::MNIST_Test::percept_MNIST();
//	BC::NN::MNIST_Test_MT::percept_MNIST();
//	BC::NN::FunctionMatcher::test();

	std::cout << "terminate success "<< std::endl;

}
