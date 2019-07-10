#### AutoDiff

BlackCat_Tensors supports auto-differentiation for all basic operations (+, -, /, *)
as well as the common trig functions:

sin, cos, tan,  
asin, acos, atan,   
sinh, cosh, tanh,  
pow, exp, sqrt  

The dx function is found in the BC::tensors namespace and is supported by all Tensors (Vector/Matrices/Cube/etc).

Example:

```cpp

#include "BlackCat_Tensors.h"

int main() {
	using BC::tanh;
	using BC::cos;
	using BC::sin;
	using BC::atanh;
	using BC::pow;

	BC::Scalar<float> x;
	BC::Scalar<float> y;

	x = .5;
	x.print();

	// dx(sin(cos(x))) == cos(cos(x)) * -sin(x)
	y = dx(sin(cos(x)));
	y.print();

	y = cos(cos(x)) * -sin(x);
	y.print();

	// dx(tanh(x)) == 1 - BC::pow(BC::tanh(x), 2)
	y = dx(tanh(x));
	y.print();

	y = 1 - pow(tanh(x), 2);
	y.print();


	std::cout << "success " << std::endl;
}

/*
Output:
	0.500000
	-0.30635
	-0.30635
	0.786448
	0.786448
	success 
*/
```
