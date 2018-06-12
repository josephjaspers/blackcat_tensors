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
	  std::string demangled = abi::__cxa_demangle(typeid(T).name(),0,0,&status);
	  return removeNS(removeNS(removeNS(demangled, "BC::"), "internal::"), "oper::");
}



int dotproduct_injection() {
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

	d = a.t();
	e = b.t();
//
	d.print();
	e.print();

	std::cout << " E" << std::endl;
	e.print();

	std::cout << " simple dot product [following should all have same value]" << std::endl;
	c = d * e;
	c.print();
	std::cout << "a.t * b.t + f" << std::endl;
	c = a.t() * b.t() + f ;
	c.print();
	std::cout << "a.t * e + f" << std::endl;
	c = a.t() * e + f;
	c.print();
//// not available to detect
////	std::cout << "-(d * b.t) + f" << std::endl;
////	c =  - (d * b.t()) + f;
////	c.print();
	std::cout << "c = d * scal(2.0f) * e + c;" << std::endl;
	c = d * scal(2.0f) * e + c;
	c.print();
	std::cout << "c = scal(2.0f) * d * e;" << std::endl;
	c = scal(2.0f) * d * e;
	c.print();
	std::cout << "c = d * e * scal(2.0f); " << std::endl;

	c = d * e * scal(2.0f); ////This is the only version that is not accounted for (it is also the least common notation)
	c.print();

	scal A(2.0f);
	scal B(2.0f);

	std::cout << " a.t * b.t" << std::endl;
	c = a.t() * b.t();
	c.print();

	std::cout << "c = a.t() * A * b.t();" << std::endl;
	A.print();
	c = a.t() * A * b.t();
	c.print();
//
	std::cout << "c = A * a.t() * b.t();" << std::endl;
	A.print();
	c = A * a.t() * b.t();
	c.print();

	std::cout << "c = a.t() * (b.t() * A);" << std::endl;
	A.print();
	c = a.t() * (b.t() * A);
	c.print();


	std::cout << "	c = a.t() * A * (b.t() * A) " << std::endl;
	A.print();
	c = a.t() * A * (b.t() * A);
	c.print();


//	using expr = std::decay_t<decltype((c =* ( BC::NN_Abreviated_Functions::g(a.t() * b.t() + c))).internal())>;

//	c =  BC::NN_Abreviated_Functions::g(a.t() * b.t() + c);

//	std::cout << type_name<expr>() << std::endl;
//	std::cout << type_name<typename BC::internal::traversal<expr>::type>() << std::endl;


	using expr2 = std::decay_t<decltype((c +=* (c*c - c*c)).internal())>;
	std::cout << type_name<expr2>() << std::endl;
	std::cout << type_name<typename BC::internal::traversal<expr2>::type>() << std::endl;
	c.print();
	c += a.t() * b.t() - a.t() * b.t();
	c.print();



//THIS DOES NOT WORK
////	std::cout << "	c = a.t() * A * (b.t() * A) + a.t() * A * (b.t() * A)" << std::endl;
////	c = a.t() * A * (b.t() * A) + a.t() * A * (b.t() * A);
//	c.print();

	return 0;
};
