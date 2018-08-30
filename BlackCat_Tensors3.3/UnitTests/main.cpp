#include <iostream>
#include "../BlackCat_Tensors.h"
#include <typeinfo>
using BC::Vector;
using BC::Matrix;
using BC::Scalar;
using BC::Cube;

using ml =  BC::CPU;	//to test cpu
//using ml = BC::GPU;	//to test gpu

using vec = Vector<float, ml>;
using mat = Matrix<float, ml>;
using scal = Scalar<float, ml>;
using cube = Cube<float, ml>;

using mat_view = BC::Matrix_View<float, ml>;
//using tensor4 = BC::Tensor4<float, ml>;
//using tesnor5 = BC::Tensor5<float, ml>;

#include "_blas_test.h"
#include "_readwrite_test.h"
#include "_shaping_test.h"
//#include <iostream>

int test() {

//various tests
//gemm_injection();

	mat a(3,3);
	a.randomize(0, 10);
	a.print();

//	(a + a).print();

	mat b(a + a);
	a.print_leading_dimensions();
	mat_view c(a);
	c.print();



	std::cout << " b is  " << std::endl;
	b.print();
	b.row(1) += b.row(1);

	b.print();


//	b = c + c;

	c += c;
	a.print();
	b = c * a;
	b.print();



	vec A(4);
	vec B(4);
	A.randomize(0, 3);
	B.randomize(0, 3);
	scal x;

	x = A * B;


	A.print();
	B.print();
	x.print();

//	b.print();
//	c.print();
//	a.print();
//	b.print();
//	b[0] = a[0];
//	b.print();
//readwrite();
//shaping();
//
//
//mat a(3,3);
//a.randomize(0,10);
//
//a.print();
////	std::cout << " a " << std::endl;
////	a.print();
//mat b(a);
//b.print();
//a.print();
//b = a;
//
//a = mat(b);
//std::cout << " b " << std::endl;
//
//b.print();
//a.randomize(0,10);
//std::cout << " a " << std::endl;
//
//a.print();
////	b = std::move(a);
//std::cout << " b " << std::endl;
//b.print();
//std::cout << " a " << std::endl;
//a.print();
//std::cout << " last call: " << std::endl;
	return 0;
}

int main() {

test();

	std::cout << " success " << std::endl;
}
