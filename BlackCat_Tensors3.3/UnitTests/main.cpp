#include <iostream>
#include "../BlackCat_Tensors.h"
#include <typeinfo>
using BC::Vector;
using BC::Matrix;
using BC::Scalar;
using BC::Cube;

using ml =  BC::CPU;		//to test cpu
//using ml = BC::GPU;	//to test gpu

using vec = Vector<float, ml>;
using mat = Matrix<float, ml>;
using scal = Scalar<float, ml>;
using cube = Cube<float, ml>;
//using tensor4 = BC::Tensor4<float, ml>;
//using tesnor5 = BC::Tensor5<float, ml>;

#include "_blas_test.h"
//#include "_readwrite_test.h"
#include "_shaping_test.h"
//#include <iostream>

int main() {


		//various tests
	gemm_injection();


	//readwrite();
	shaping();
	mat a(3,3);
	a.randomize(0,10);

	a.print();
//	std::cout << " a " << std::endl;
//	a.print();
	mat b = a;
	b.print();
	a.print();
	b = a;

	a = mat(b);
//	std::cout << " b " << std::endl;
//
//	b.print();
//	a.randomize(0,10);
//	std::cout << " a " << std::endl;
//
//	a.print();
////	b = std::move(a);
//	std::cout << " b " << std::endl;
//
//	b.print();
//	std::cout << " a " << std::endl;
//
//	a.print();


	std::cout << " success " << std::endl;
}
