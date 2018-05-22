#include <iostream>
#include "../BlackCat_Tensors.h"

using BC::Vector;
using BC::Matrix;
using BC::Scalar;
using BC::Cube;

using ml = BC::CPU;
//using ml = BC::GPU;

using vec = Vector<float, ml>;
using mat = Matrix<float, ml>;
using scal = Scalar<float, ml>;
using cube = Cube<float, ml>;
using tensor4 = BC::Tensor4<float, ml>;
using tesnor5 = BC::Tensor5<float, ml>;

#include "_correlation_test.h"
#include "_dotproducts_test.h"
#include "_readwrite_test.h"
#include "_shaping_test.h"
#include "_speed_benchmark.h"

int main() {

	correlation();
	dotproducts();
	readwrite();
	shaping();


	std::cout << " success  main" << std::endl;

}
