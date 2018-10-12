/*
 * Shaping_Test.h
 *
 *  Created on: Oct 6, 2018
 *      Author: joseph
 */

#ifndef SHAPING_TEST_H_
#define SHAPING_TEST_H_

using matrix = BC::Matrix<double>;


void shaping_test() {

	matrix a(4,4);
	a.zero();

	for (int i = 0; i < a.size(); ++i){
		a(i) = i;
	}

	a.print();
	a[0].print();
	a[1].print();
	a[{1,3}].print(); //a[1:3]

	a.row(0).print();
	a.row(1).print();
	a.diag().print();
	a.diag(-1).print();
	a.diag(2).print();

	reshape(a)(4,2,2).print();
	chunk(a, 1, 1)(2,2).print();

	chunk(a, 1,1)(2,2) += chunk(a,1,1)(2,2);
	a.print();

//	format_as(a, 0,2,1).print();

	BC::Cube<double> c(3,3,3);
	for (int i = 0; i < c.size(); ++i){
		c(i) = i;
	}
	c.print();
	format_as(c, 0, 2, 1).print();
	format_as(c, 0, 2, 1)[0].print();

}



#endif /* SHAPING_TEST_H_ */
