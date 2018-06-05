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

#include "_correlation_test.h"
#include "_dotproducts_test.h"
#include "_readwrite_test.h"
#include "_shaping_test.h"
#include <iostream>
//#include "_speed_benchmark.h"
//#include "_d1_xcorr_test.h"

#include <iostream>


#include <cxxabi.h>

std::string removeNS( const std::string & source, const std::string & namespace_ )
{
    std::string dst = source;
    size_t position = source.find( namespace_ );
    while ( position != std::string::npos )
    {
        dst.erase( position, namespace_.length() );
        position = dst.find( namespace_, position + 1 );
    }
    return dst;
}

template<class T>
std::string type_name() {
	int status;
	  std::string demangled = abi::__cxa_demangle(typeid(T).name(),0,0,&status);
	  return removeNS(removeNS(removeNS(demangled, "BC::"), "internal::"), "function::");
}



template<class T>
auto g(const T& tensor) {
	return tensor.un_expr([](auto value) { return 1 / (1 + 2.7182 * value); });
}

int main() {

	//various tests
//	correlation();
	dotproducts();
	readwrite();
	shaping();

	mat x(2,2);
	mat y(2,2);
	mat z(2,2);
	z.zero();
	for (int i = 0; i < 4; ++i) {
		x(i) = i;
		y(i) = i + 4;
	}

	x.print();
	y.print();

	using t = decltype((x * y).internal());
	std::cout << "evaluate - BLAS detection - " << BC::internal::INJECTION<t>() << std::endl;
	std::cout << type_name<t>() << std::endl << std::endl;


//	using U = decltype((x =* (x * x)).internal());
		using U = decltype((x =* (x * x + y)).internal());
		auto function = (z =* (x * x + y)).internal();
	std::cout << "evaluate - BLAS detection - " << BC::internal::INJECTION<U>() << std::endl;
	std::cout << type_name<U>() << std::endl << std::endl;

	using adjusted = BC::internal::injection_t<U, decltype(x.internal())>;//typename BC::internal::injector<std::decay_t<U>>::template type<decltype(x.internal())>;
	using tens = BC::tensor_of_t<adjusted::DIMS(), adjusted, ml>;


	std::cout << type_name<adjusted>() << std::endl << std::endl;

	//	cube c(4,3,3);//output
//	mat b(5,5);//img
//	b.randomize(0,1);
//	b.print();
//	c.zero();
	adjusted fixed(function, z.internal());

	for (int i = 0; i < 4; ++i) {
		std::cout << fixed[i] << std::endl;
	}

	auto var = tens(adjusted(function, z.internal()));

	mat output = var;
	output.print();
	std::cout << " success  main" << std::endl;

}
