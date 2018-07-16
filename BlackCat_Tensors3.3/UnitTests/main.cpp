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
//	gemm_injection();
//	readwrite();
//	shaping();

	/*
	 * JOSEPH YOU WAKE UP TOMORROW NAD WONDER WHAT I WAS DOING HORIZONTAL ACCESS IS SHITTY STILL FIX IT
	 */

	cube a(2,4,6);

	for (int i = 0; i < a.size(); ++i){
		a(i) = i;
	}

	mat b(4, 6);
	vec c(6);

	c = 1;

	a.print();
	b.zero();

	std::cout << " success " << std::endl;
}
