#include "../BlackCat_Tensors/BlackCat_Tensors.h"
#include "../BlackCat_NeuralNetworks/BlackCat_GPU_NeuralNetworks.h"

#include "MNIST_test.h"
//#include "Shaping_Test.h"
//#include "Recurrent_Test.h"
int main() {

//	shaping_test();

//	BC::NN::MNIST_Test::percept_MNIST();
//	BC::NN::rec_MNIST_test::percept_MNIST();

//
	BC::Matrix<float, BC::GPU> a(3,3);
	BC::Matrix<float, BC::GPU> b(3,3);

//
////	a = a * 2 * b;
////	b = 2 * a * b;
////	b = a * b * 2;
//
//	a.print();
//
	for (int i = 0; i < 9; ++i) {
		a(i) = i;
		b(i) = i;
	}
////
//
	a = 1 - a;
	a.randomize(0, 10);
	a.print();
	a.print();
	a = 3;
	a.print();
//	a = 2;
	a.fill(2);
	a.print();
	a.zero();
	a.print();

	a = 1 - a;
	a.print();
//
//	a = a*2*b;
//	a = a * b * 2;
//	a = 2 * a * b;
	std::cout << " success " <<std::endl;
//	a = (a-b) + a * b;
//
//
//	BC::Matrix<float> mat_c = a * b;
//
//	mat_c.print();
//	BC::Vector<float> c(3);
////	c = (a * b)[0];
//
//	BC::Scalar<float> d;
//
////	c = (a * b)[0];
////	d = (a * b)[0][0];
//d.print();
//c.print();

return 0;
}
