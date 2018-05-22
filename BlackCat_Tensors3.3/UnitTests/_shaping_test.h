/*
 * _shaping_test.h
 *
 *  Created on: May 22, 2018
 *      Author: joseph
 */

#ifndef SHAPING_TEST_H_
#define SHAPING_TEST_H_

#include <iostream>
#include "../BlackCat_Tensors.h"

using BC::Vector;
using BC::Matrix;
using BC::Cube;

int shaping() {

	//parameter oacks + lamdas not supported by CUDA yet
#ifndef __CUDACC__
	std::cout << " --------------------------------------SHAPING (RESHAPE, AND CHUNK)--------------------------------------" << std::endl;



	Cube<float, BC::CPU> m2(4,4, 3);
	m2.randomize(0,100);
	std::cout << " original " << std::endl;
	m2.print();

	std::cout << " reshape (16,3) " << std::endl;
	reshape(m2)(16,3).print();

	std::cout << " chunk at (3,2,1) dims (2,3)" << std::endl;
	chunk(m2)(3,2,1)(2,3).print();


	std::cout << "reshape (2,3) chunk to (3,2)" << std::endl;
	auto ch = chunk(m2)(3,2,1)(2,3);
	reshape(ch)(3,2).print();

#endif
	return 0;
}



#endif /* SHAPING_TEST_H_ */
