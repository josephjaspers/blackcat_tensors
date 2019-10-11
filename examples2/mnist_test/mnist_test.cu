/*
 * MNIST_Test.cpp
 *
 *  Created on: Jul 15, 2019
 *      Author: joseph
 */

#ifndef MNIST_TEST_CU_
#define MNIST_TEST_CU_

#include "mnist_test.h"	      //Add BlackCat_Tensors/examples/ to path
#include <iostream>

int main(int argc, const char* args[]) {
	if (argc != 2) {
		std::cout << "Please supply the path to the MNIST dataset " <<
				"(default location is ../BlackCat_Tensors/examples/mnist_test/ )" << std::endl;
		return 1;
	}
	std::cout << "Dataset location: " << args[1] << std::endl;

	percept_MNIST(BC::device_tag(), args[1]);
}

#endif /* MNIST_TEST_CU_ */
