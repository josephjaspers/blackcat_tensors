#include <iostream>
#include "../BlackCat_Tensors.h"

using BC::Vector;
using BC::Matrix;
using BC::Scalar;
using BC::Cube;

//using ml = BC::CPU;
using ml = BC::GPU;

using vec = Vector<float, ml>;
using mat = Matrix<float, ml>;
using scal = Scalar<float, ml>;
using cube = Cube<float, ml>;
using tensor4 = BC::Tensor4<float, ml>;
using tesnor5 = BC::Tensor5<float, ml>;


//std::vector<unsigned int> data_type;
//using ary = std::vector<unsigned int>;

#include "_correlation_test.h"
#include "_dotproducts_test.h"
#include "_readwrite_test.h"
#include "_shaping_test.h"
#include <iostream>
//#include "_speed_benchmark.h"
//#include "_d1_xcorr_test.h"

int main() {

	//various tests
//	correlation();
	dotproducts();
	readwrite();
	shaping();

//	cube c(4,3,3);//output
//	mat b(5,5);//img
//	b.randomize(0,1);
//	b.print();
//	c.zero();


	std::cout << " success  main" << std::endl;

}
