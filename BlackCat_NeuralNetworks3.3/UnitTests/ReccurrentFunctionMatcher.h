#ifndef RECURRENT_TEST
#define RECURRENT_TEST
#include "../BlackCat_NeuralNetworks.h"
#include <fstream>
#include <iostream>
#include <string>
#include <omp.h>
#include <random>
using BC::NN::vec;
using BC::NN::scal;
using BC::NN::mat;

namespace BC {
namespace NN {
namespace FunctionMatcher {
typedef std::vector<mat> data;

//CPU only
mat fixed_sin_set(int n) {
	mat sins(1, n);

//	std::cout << " length = " << n << std::endl;

	float seed = rand();
	while (seed > 100)
		seed /= 10;

	for (int i = 0; i < n; ++i) {
//		std::cout << " seed = " << seed << " sin = " << std::sin(seed) << std::endl;
		seed += 3.14 / 8;
		sins(i) = (std::sin(seed) + 1) / 2;
	}

//	sins.print();
	return sins;
}



int test() {

	const int TRAINING_ITERATIONS = 100;
	const int NUMB_EXAMPLES = 1000;
	const int TESTS  = 10;
	NeuralNetwork<FeedForward, GRU, FeedForward> network(1, 100, 40, 1);
	network.setLearningRate(.003);

	data inputs(NUMB_EXAMPLES);
	for (int i = 0; i < NUMB_EXAMPLES; ++i){
		int length = (rand() % 10) + 3;
		inputs[i] = fixed_sin_set(length);
	}

	//Train
	float t;
	t = omp_get_wtime();
	printf("\n Calculating... BC_NN training time \n");
	std::cout << " training..." << std::endl;

	for (int i = 0; i < TRAINING_ITERATIONS; ++i) {
		if (i % 10 == 0)
		std::cout << " iteration =  " << i << std::endl;
		for (int j = 0; j < inputs.size(); ++j) {
			for (int l = 0; l < inputs[j].cols()-1; ++l)
				network.forwardPropagation(inputs[j][l]);
//
			for (int l = inputs[j].cols()-1; l > 0; --l)
			network.backPropagation(inputs[j][l]);
//
			network.updateWeights();
			network.clearBPStorage();
		}
	}

//	data test_vars(TESTS);
//	for (int i = 0; i < TESTS; ++i){
//		int length = rand() * 10;
//		test_vars[i] = fixed_sin_set(length);
//	}
//
//
	for (int i = 0; i < 15; ++i) {
		std::cout << std::endl << std::endl;
		std::cout << " set ------------------" << std::endl;
		for (int j = 0; j < inputs[i].cols(); ++j) {
//			std::cout << "input : ";
//			inputs[i][j].print();

//			std::cout << "  exp : ";
//			inputs[i][j + 1].print();
//
//			std::cout << "  out : ";
			vec x = inputs[i][j + 1] - network.forwardPropagation_Express(inputs[i][j]);
			std::cout << " error = "; x.print();
		}
	}
}


}
}
}
#endif
