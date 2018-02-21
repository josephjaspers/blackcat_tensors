#include <iostream>
#include "../BlackCat_Tensors.h"
#include "SpeedTests.h"
using BC::Vector;
using BC::Matrix;
using BC::Scalar;

template<class T>
T copy(const T& t) { return t; }

auto dp_test () {

	Matrix<double> a(3, 2);
	Matrix<double> b(2 ,3);
	Matrix<double> d(2, 3);
	Matrix<double> e(3, 2);
	Matrix<double> c(2, 2);

	for (int i = 0; i < 6; ++i)  {
		b.data()[i] = i + 7;
		a.data()[i] = i + 1;
	}

	std::cout << std::endl;





	d = a.t();
	e = b.t();


	a.print();
	b.print();
	d.print();
	e.print();



	std::cout << " post print  " << std::endl;

//	Scalar<double> C(2);

//	Vector<double> D_(6);

	std::cout << " simple dot product " << std::endl;
	c = d * e;
	///all the permutations of optimized dotproduct
	c = a.t() * b.t();
	c = a.t() * e;
	c = d * b.t();
	c = d * Scalar<double>(2) * e;
	c = Scalar<double>(2) * d * e;
	c = d * e * Scalar<double>(2); ////This is the only version that is not accounted for (it is also the least common notation)
	c = d * Scalar<double>(2) * e;

	c.print();


	Scalar<double> A(2);
	Scalar<double> B(2);

	std::cout << " dot product " << std::endl;

	c[0].print();
	c.print();

	c = a.t() * b.t();
	std::cout << " dot product 222 " << std::endl;

	A.print();

	c = a.t() * A * (b.t() * A);

	c.print();

	std::cout << "done  copy " <<
			std::endl;


	std::cout << "successsasd " << std::endl;
//	return 0;
//	return c_copy;
}


int VectorPointwise() {

std::cout << "Vector test " <<std::endl;


	Vector<double> a = {5};
	Vector<double> b = {5};
	Vector<double> c = {5};


	a[0];
	a[0].print();
	std::cout << " getting size" <<std::endl;


	std::cout << b.size() << " getting size" <<std::endl;
	b.printDimensions();
	a.printDimensions();

//	a.printDetails();


////	b.print();
	std::cout << "Randomizing " << std::endl;
	b.randomize(-3, 5);
	c.randomize(0, 5);
//

	a.data()[0] = 123123;
	std::cout << " B " << std::endl;
	b.print();
	std::cout << " C" << std::endl;
	c.print();
//
//
	std::cout << " c= b.t" << std::endl;
//
	c = b.t();
	c.print();
//
	std::cout << " B + C" << std::endl;
	a = b + c;
	a.print();

	a.data().array[0] = 3243;
	a.print();
//
	std::cout << " a + a + a" << std::endl;
	a = a + a + a;

	a.print();

	std::cout << " B ** C" << std::endl;
	b.print();
	c.print();

	a = b ** c;
	a.print();


	return 0;
}

int MatrixPointwise() {

std::cout << "Matrix test " <<std::endl;

	Matrix<double> a(4,3);
	Matrix<double> b(4,3);
	Matrix<double> c(4,3);


	Matrix<double> z(5,5 );
	Vector<double> d(3);


	*d;
	Vector<double> e(4);

	b.randomize(0, 10);
	c.randomize(0, 10);

	b.print();
	c.print();

	std::cout << " c= b.t" << std::endl;

//	c = b.t();

	std::cout << "A = b + c" << std::endl;
	a = b + c;

	Scalar<double> sc(10000);

	a.print();

	a = a + sc;
	a.print();




	std::cout << " adding array " << std::endl;

//c.printDetails();
//d.printDetails();
	e = (c * d);

	std::cout << " success " << std::endl;

	return 0;
}

struct A {
	int* alpha;

	operator int*() { return alpha; }
};

template<class,class>
struct same  { static constexpr bool value = false; };

template<class t>
struct same<t,t>  { static constexpr bool value = true; };





int main() {
//
	speedTestDelayedEval<128,     100000>();
	speedTestDelayedEval<256,     100000>();
	speedTestDelayedEval<512,     100000>();
	speedTestDelayedEval<1024,    100000>();
	speedTestDelayedEval<2048,    100000>();
	speedTestDelayedEval<5096,    100000>();
	speedTestDelayedEval<10000,   100000>();
	speedTestDelayedEval<20000,   100000>();
	speedTestDelayedEval<40000,   100000>();
	speedTestDelayedEval<80000,   100000>();
	speedTestDelayedEval<100000,  100000>();
//	MatrixPointwise();
//	VectorPointwise();
//
//		dp_test();

	std::cout << " success  main"<< std::endl;


	return 0;
}
