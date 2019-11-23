/*
 * MNIST_Test.cpp
 *
 *  Created on: Jul 15, 2019
 *      Author: joseph
 */

#ifndef MNIST_TEST_CPP_
#define MNIST_TEST_CPP_

#include "mnist_test.h"	      //Add BlackCat_Tensors/examples/ to path
#include <iostream>

int main(int argc, const char* args[]) {
	std::string mnist_filepath;

	if (argc > 1) {
		mnist_filepath = args[1];
	} else {
		mnist_filepath = "../datasets/mnist_train.csv";
	}
#ifdef __CUDACC__
	percept_MNIST(BC::device_tag(), mnist_filepath);
#else
	percept_MNIST(BC::host_tag(), mnist_filepath);
#endif
}

#endif /* MNIST_TEST_CPP_ */
