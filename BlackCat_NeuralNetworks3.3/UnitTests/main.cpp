#include "MNIST_test.h"
#include "MNIST_test_MT.h"
#include "CharacterPrediction.h"
#include "ReccurrentFunctionMatcher.h"
#include "Data/fixed_TheRaven.h"
#define BC_DISABLE_OPENMP

int main() {

//	BC::NN::MNIST_Test::percept_MNIST();
//	BC::NN::MNIST_Test_MT::percept_MNIST();
//	BC::NN::FunctionMatcher::test();
	BC::NN::Word_Test::test(TheRaven, 1);
	BC::NN::Word_Test::test(TheRaven, 2);
	BC::NN::Word_Test::test(TheRaven, 3);

	std::cout << "terminate success "<< std::endl;

}
