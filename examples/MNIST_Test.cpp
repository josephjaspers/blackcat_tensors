/*
 * MNIST_Test.cpp
 *
 *  Created on: Jul 15, 2019
 *      Author: joseph
 */

#ifndef MNIST_TEST_CPP_
#define MNIST_TEST_CPP_

#include "BlackCat_Tensors.h" //Add BlackCat_Tensors/include/ to path
#include "MNIST_Test.h"	      //Add BlackCat_Tensors/examples/ to path

int main() {
#ifdef __CUDACC__
	BC::nn::percept_MNIST(BC::device_tag());
#else
	BC::nn::percept_MNIST(BC::host_tag());
#endif
}

#endif /* MNIST_TEST_CPP_ */
