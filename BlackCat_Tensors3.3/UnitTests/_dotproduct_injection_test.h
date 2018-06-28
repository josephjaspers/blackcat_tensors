#include <iostream>
#include "../BlackCat_Tensors.h"
#include "../Extensions/NN_Functions.h"

using BC::Vector;
using BC::Matrix;

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
//	return abi::__cxa_demangle(typeid(T).name(),0,0,&status);;
	  std::string demangled = abi::__cxa_demangle(typeid(T).name(),0,0,&status);
	  return removeNS(removeNS(removeNS(demangled, "BC::"), "internal::"), "oper::");
}



int gemm_injection() {
	std::cout << " --------------------------------------DOTPRODUCTS--------------------------------------" << std::endl;

	mat a(3, 2);
	mat b(2 ,3);
	mat d(2, 3);
	mat e(3, 2);
	mat c(2, 2);

	mat f(2, 2);
	mat abs(10,10);

	abs.print_dimensions();
	f.zero();
	std::cout << " param " << std::endl;
	a.print_dimensions();

	for (int i = 0; i < 6; ++i)  {
		a(i) = i + 7;
		b(i) = i + 1;
	}
	f(0) = 10;
	f(3) = 20;


	a.print();
	b.print();
	f.print();

	std::cout << "d = a.t" << std::endl;
	d = a.t() + b - b;
	d.print();

	std::cout << "e = b.t" << std::endl;
	e = b.t() + a - a;
	e.print();

	std::cout << " c = d * e[following should all have same value]" << std::endl;
	c.zero();
	c = d * e;
	c.print();
	std::cout << "c = a.t * b.t + f" << std::endl;
	c.zero();
	c = a.t() * b.t() + f ;

	c.print();
	std::cout << "c = a.t * e + f" << std::endl;
	c.zero();
	c = a.t() * e + f;
	c.print();
//// not available to detect
////	std::cout << "-(d * b.t) + f" << std::endl;
////	c =  - (d * b.t()) + f;
////	c.print();
	std::cout << "c = d * scal(2.0f) * e + f;" << std::endl;
	c.zero();
	c = d * scal(2.0f) * e + f;

	std::cout << type_name<decltype(d * scal(2.0f) * e + f)>() << std::endl;
	std::cout << decltype(d * scal(2.0))::DIMS() << " GET DIMS " << std::endl;
//	std::cout << (d * scal(2.0)).dims() << " GET DIMS " << std::endl;

	c.print();
	std::cout << "c = scal(2.0f) * d * e;" << std::endl;
	c.zero();
	c = scal(2.0f) * d * e;
	c.print();
//	std::cout << "c = d * e * scal(2.0f); " << std::endl;
//	c.zero();
//	c = d * e * scal(2.0f); ////This is the only version that is not accounted for (it is also the least common notation)
//	c.print();

	scal A(2.0f);
	scal B(2.0f);

	std::cout << " a.t * b.t" << std::endl;
	c.zero();
	c = a.t() * b.t();
	c.print();

	std::cout << "c = a.t() * A * b.t();" << std::endl;
	c.zero();
	A.print();
	c = a.t() * A * b.t();
	c.print();
//
	std::cout << "c = A * a.t() * b.t();" << std::endl;
	c.zero();
	A.print();
	c = A * a.t() * b.t();
	c.print();

	std::cout << "c = a.t() * (b.t() * A);" << std::endl;
	c.zero();
	A.print();
	c = a.t() * (b.t() * A);
	c.print();


	std::cout << "	c = a.t() * A * (b.t() * A) " << std::endl;
	c.zero();
	A.print();
	c = a.t() * A * (b.t() * A);
	c.print();


	std::cout << "	c =  - (a.t() * A * (b.t() * A))" << std::endl;
	c.zero();
	A.print();
	c = (a.t() * A * (b.t() * A));

	std::cout << "	c = (a.t() * b.t()) % (a.t() * b.t()); " << std::endl;
	c.zero();
	c = (a.t() * b.t()) % (a.t() * b.t());
	c.print();

	mat F(2,2);

	c.print();
	F = c * (a.t() * b.t());
	F.print();

	c.alias() = c * (a.t() * b.t());
	c.print();

	std::cout << " no alias  " << std::endl;
	c = c * (a.t() * b.t());
	c.print();

	vec x(2);
	vec y(2);

	y.print();


	x = 1;


	std::cout << " this should be gemv call " << std::endl;
	x.print();
	c.print();
	y -= c * x;

	y.print();


	return 0;
};
