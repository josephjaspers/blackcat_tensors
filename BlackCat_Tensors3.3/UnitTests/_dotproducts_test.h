#include <iostream>
#include "../BlackCat_Tensors.h"

using BC::Vector;
using BC::Matrix;

int gemms() {
	std::cout << " --------------------------------------DOTPRODUCTS--------------------------------------" << std::endl;

	mat a(3, 2);
	mat b(2 ,3);
	mat d(2, 3);
	mat e(3, 2);
	mat c(2, 2);
	mat abs(10,10);

	abs.print_dimensions();

	std::cout << " param " << std::endl;

	a.print_dimensions();

	for (int i = 0; i < 6; ++i)  {
		a(i) = i + 7;
		b(i) = i + 1;
	}


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
	std::cout << " dot product -- transpose, transpose" << std::endl;
	c = a.t() * b.t();
	c.print();
	std::cout << " dot product -- transpose, regular" << std::endl;
	c = a.t() * e;
	c.print();
	std::cout << " dot product -- regular, transpose" << std::endl;
	c = d * b.t();
	c.print();
	std::cout << " dot product -- scalar, regular" << std::endl;
	c = d * scal(2.0f) * e + c;
	c.print();
	std::cout << " dot product -- scalar, regular" << std::endl;
	c = scal(2.0f) * d * e;
	c.print();
	std::cout << " dot product -- regular,  scalar " << std::endl;

	c = d * e * scal(2.0f); ////This is the only version that is not accounted for (it is also the least common notation)
	c.print();

	scal A(2.0f);
	scal B(2.0f);

	std::cout << " dot product -- trans, trans " << std::endl;
	c = a.t() * b.t();
	c.print();

	std::cout << " dot product -- trans scalar, trans " << std::endl;
	A.print();
	c = a.t() * A * b.t();
	c.print();
//
	std::cout << " dot product -- scalar trans, trans " << std::endl;
	A.print();
	c = A * a.t() * b.t();
	c.print();

	std::cout << " dot product -- trans, trans scalar " << std::endl;
	A.print();
	c = a.t() * (b.t() * A);
	c.print();

	std::cout << " dot product -- trans scalar, trans scalar " << std::endl;
	A.print();
	c = a.t() * A * (b.t() * A);
	c.print();


	return 0;
};
