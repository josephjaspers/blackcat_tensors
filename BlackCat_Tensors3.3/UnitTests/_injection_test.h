/*
 * _injection_test.h
 *
 *  Created on: Jun 5, 2018
 *      Author: joseph
 */

#ifndef INJECTION_TEST_H_
#define INJECTION_TEST_H_

#include <iostream>
#include "../BlackCat_Tensors.h"

using namespace NN_Functions;
using BC::Vector;
using BC::Matrix;

void injection() {

	mat x(2,2);
	mat y(2,2);
	mat z(2,2);
	z.zero();
	for (int i = 0; i < 4; ++i) {
		x(i) = i;
		y(i) = i + 4;
	}

	x.print();
	y.print();

	using t = decltype((x * y).internal());
	std::cout << "evaluate - BLAS detection - " << BC::internal::INJECTION<t>() << std::endl;
	std::cout << type_name<t>() << std::endl << std::endl;


//	using U = decltype((x =* (x * x)).internal());
		using U = decltype((x =* (x * x + y)).internal());
		auto function = (z =* (x * x + y)).internal();
	std::cout << "evaluate - BLAS detection - " << BC::internal::INJECTION<U>() << std::endl;
	std::cout << type_name<U>() << std::endl << std::endl;

	using adjusted = BC::internal::injection_t<U, decltype(x.internal())>;//typename BC::internal::injector<std::decay_t<U>>::template type<decltype(x.internal())>;
	using tens = BC::tensor_of_t<adjusted::DIMS(), adjusted, ml>;


	std::cout << type_name<adjusted>() << std::endl << std::endl;
	//	cube c(4,3,3);//output
//	mat b(5,5);//img
//	b.randomize(0,1);
//	b.print();
//	c.zero();
	adjusted fixed(function, z.internal());

	for (int i = 0; i < 4; ++i) {
		std::cout << fixed[i] << std::endl;
	}

	auto var = tens(adjusted(function, z.internal()));

	mat output = var;
	output.print();

}



#endif /* INJECTION_TEST_H_ */
