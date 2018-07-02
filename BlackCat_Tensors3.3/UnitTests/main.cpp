#include <iostream>
#include "../BlackCat_Tensors.h"
#include <typeinfo>
using BC::Vector;
using BC::Matrix;
using BC::Scalar;
using BC::Cube;

using ml = BC::CPU;		//to test cpu
//using ml = BC::GPU;	//to test gpu

using vec = Vector<float, ml>;
using mat = Matrix<float, ml>;
using scal = Scalar<float, ml>;
using cube = Cube<float, ml>;
using tensor4 = BC::Tensor4<float, ml>;
using tesnor5 = BC::Tensor5<float, ml>;

#include "_blas_test.h"
#include "_readwrite_test.h"
#include "_shaping_test.h"
#include <iostream>

int main() {

	//various tests
//	correlation();
//	gemms();
	gemm_injection();
//	readwrite();
//	shaping();


	mat a(2,2);
	vec x(2);
	vec y(2);

	a = x * y.t();


	mat b(x);
	b.print_dimensions();
	b.print_leading_dimensions();

	std::cout << " success  main" << std::endl;

}
