#include <iostream>
#include "../BlackCat_Tensors.h"
#include <typeinfo>
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


//std::vector<unsigned int> internal_type;
//using ary = std::vector<unsigned int>;

//#include "_correlation_test.h"
//#include "_gemms_test.h"
#include "_dotproduct_injection_test.h"

//#include "_readwrite_test.h"
//#include "_shaping_test.h"
#include <iostream>
//#include "_speed_benchmark.h"
//#include "_d1_xcorr_test.h"

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

//	using chunk_t = decltype(chunk(w)(0,0)(0,0).internal());
//
//	using core = std::decay_t<decltype(w.internal())>;
//	using expr = std::decay_t<decltype((chunk(w)(0,0)(0,0) =* (abs(w * w + w))).internal())>;
//
//
//	using sub_t = BC::internal::traversal<expr>::type;
//
//	std::cout << type_name<expr>() << std::endl;
//	std::cout << type_name<sub_t>() << std::endl;
//
//


	std::cout << " success  main" << std::endl;

}
