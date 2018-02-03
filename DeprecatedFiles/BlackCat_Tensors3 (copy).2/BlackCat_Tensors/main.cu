#include <iostream>
#include "BlackCat_Tensors.h"
using BC::Vector;
using BC::Matrix;
using BC::Scalar;

int VectorPointwise() {

std::cout << "Vector test " <<std::endl;

	Vector<double> a = {5};
	Vector<double> b = {5};
	Vector<double> c = {5};

	std::cout << " getting size" <<std::endl;


	std::cout << b.size() << " getting size" <<std::endl;
	b.randomize(-3, 5);
	c.randomize(0, 5);

	std::cout << " B " << std::endl;
	b.print();
	std::cout << " C" << std::endl;
	c.print();

	std::cout << " B + C" << std::endl;
	a = b + c;
	a.print();

	std::cout << " B - C" << std::endl;
	a = b - c;


	std::cout << " B ** C" << std::endl;
	b.print();
	c.print();

	a = b ** c;
	a.print();


	std::cout << "b[0]" << std::endl;
	b[0].print();


	std::cout << "a" << std::endl;
	a.print();


	std::cout << "a += b[0]" << std::endl;
	a += b[0];
	a.print();


	std::cout << " success " << std::endl;
	return 0;


}
//
int MatrixPointwise() {

std::cout << "Matrix test " <<std::endl;

	Matrix<double> a(4,3);
	Matrix<double> b(4,3);
	Matrix<double> c(4,3);


	Matrix<double> z(5,5 );
	Vector<double> d(3);

	*d;;

//	d = d *d *;
	Vector<double> e(4);

	b.randomize(0, 10);
	c.randomize(0, 10);

	b.print();
	c.print();

	std::cout << "A = b + c" << std::endl;
	a = b + c;

	a.print();



	a[0][1].print();

	std::cout << " adding array " << std::endl;
b.print();
b += a[0][1];
b.print();
	e = (c * d);


	e.print();
	e[1] = 2;
	z.randomize(1,2);

	e[1].print();

	z = z.t() * e[1] * z;

	z.print();
	std::cout << " success " << std::endl;

	return 0;
}

int gpu() {



	Vector<float, BC::GPU> a(10);
	Vector<float, BC::GPU> b(10);
	Vector<float, BC::GPU> c(10);

	a.randomize(0, 3);
	b.randomize(0, 3);

	a.print();
	b.print();

   c = (a + b);// + (b % B);

   c.print();

	std::cout << " success " << std::endl;
	return 0;
}

int dp_test () {

	Matrix<double> a = {3, 2};
	Matrix<double> b = {2 ,3};

	for (int i = 0; i < 6; ++i)  {
		a.data()[i] = i + 7;
		b.data()[i] = i + 1;
	}



	std::cout << std::endl;

	Matrix<double> d = {2, 3};
	Matrix<double> e = {3, 2};

//	a.print();
//	b.print();

	d = a.t();
	e = b.t();

	d.print();
	e.print();
//	for (int i = 0; i < 6; ++i)  {
//		std::cout << d.data()[i] << " ";
//	}
//	std::cout << std::endl;
//	for (int i = 0; i < 6; ++i)  {
//		std::cout << e.data()[i] << " ";
//	}

	Matrix<double> c = {2, 2};

	c = a.t() * b.t();//a.t() * b.t();

	c.print();



	c.print();


	std::cout << "success " << std::endl;
	return 0;
}

int main() {
//	gpu();
	dp_test();
//	VectorPointwise();
//	MatrixPointwise();

	return 0;
}
