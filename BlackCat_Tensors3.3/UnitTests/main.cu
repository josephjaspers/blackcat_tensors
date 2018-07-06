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



//
	mat a(4,4);

	for (int i = 0; i < a.size(); ++i){
		a(i) = i;
	}
	a.print();

	mat k(2,2);
	for (int i = 0; i < k.size(); ++i){
		k(i) = i;
	}

	k.print();

	mat c(3,3);

	c = a.conv<2>(k);
	c.print();

	std::cout << " success " << std::endl;

}
