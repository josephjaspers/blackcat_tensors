#include "MNIST_test.h"
#include "MNIST_test_MT.h"
#include "CharacterPrediction.h"
#include "ReccurrentFunctionMatcher.h"
#define BC_DISABLE_OPENMP

#include "Data/fixed_ThePit.h"
#include "Data/fixed_TheCaskOfAmontillado.h"
#include "Data/fixed_AngelOfTheOdd.h"
#include "Data/fixed_TheRaven.h"

int main() {

//	BC::NN::MNIST_Test::percept_MNIST();
////
//	BC::NN::MNIST_Test_MT::percept_MNIST();


//	for (int i = 0; i < 100; ++i) {

//	BC::NN::Word_Test::test(ThePit);
//	BC::NN::Word_Test::test(TheCaskOfAmontillado);
//	BC::NN::Word_Test::test(AngelOfTheOdd);
	BC::NN::Word_Test::test(TheRaven);
//	}

	//	BC::NN::FunctionMatcher::test();

	std::cout << "terminate success "<< std::endl;

}
