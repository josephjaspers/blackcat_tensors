/*
 * readwrite_test.h
 *
 *  Created on: May 22, 2018
 *      Author: joseph
 */

#ifndef READWRITE_TEST_H_
#define READWRITE_TEST_H_
#include <iostream>
#include "../BlackCat_Tensors.h"

using BC::Vector;
using BC::Matrix;

int readwrite() {
	std::cout << " --------------------------------------READWRITE--------------------------------------" << std::endl;

	mat d(10,10);
	d.randomize(0,100);
	std::cout << " matrix is " << std::endl;
	d.print();

	std::cout << " trying to write " <<std::endl;
	std::ofstream os("save.txt");
	d.write(os);
	os.close();


	std::cout << " trying to read" << std::endl;
	std::ifstream is("save.txt");

	mat readM(d.size());
	readM.read(is);
	readM.print();
	is.close();

	return 0;
}



#endif /* READWRITE_TEST_H_ */
