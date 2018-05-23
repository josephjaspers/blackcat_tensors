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
#include "_d1_xcorr_test.h"
int main() {

	//various tests
//	correlation();
//	dotproducts();
//	readwrite();

	shaping();
//	for (int i = 1; i < 100; i *= 2)

	cube c(3, 5,4);
	std::cout << c.data().scal_index(1,2,3) << std::endl;
	std::cout << c.data().scal_index(BC::array(1,2,3)) << std::endl;

	std::cout << c.data().scal_index_reverse(3,2,1) << std::endl;
	std::cout << c.data().scal_index_reverse(BC::array(3,2,1)) << std::endl;



	std::cout << " success  main" << std::endl;

}
