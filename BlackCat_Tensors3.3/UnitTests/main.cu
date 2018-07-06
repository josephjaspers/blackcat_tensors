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


//	mat i2c(4,9);
//	ml::img2col(reshape(i2c)(2,2,3,3).internal(), a);
//	i2c.print();

	mat c(3,3);
	ml::conv2d(c.internal(), a.internal(), k.internal());


	c.print();

	vec d(3);
	d.zero();
	d += c;
	d.print();

	d.zero();
	d += c.t();
	d.print();
	std::cout << " success " << std::endl;

}
