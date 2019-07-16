/*
 * MNIST_Test.cpp
 *
 *  Created on: Jul 15, 2019
 *      Author: joseph
 */

#ifndef MNIST_TEST_CU_
#define MNIST_TEST_CU_

#include "BlackCat_Tensors.h" //Add BlackCat_Tensors/include/ to path
#include "MNIST_Test.h"	      //Add BlackCat_Tensors/examples/ to path

int main() {
	BC::nn::percept_MNIST(BC::device_tag());
}

#endif /* MNIST_TEST_CU_ */
