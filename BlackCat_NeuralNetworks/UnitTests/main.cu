#include "MNIST_test.h"
//#include "MNIST_Auto_Encoder.h"
#include "BlackCat_Tensors.h"
#include <iostream>
int main() {
	BC::NN::MNIST_Test::percept_MNIST();
//	BC::NN::MNIST_Auto_Encoder_Test::percept_MNIST();

//	BC::Scalar<float, BC::GPU> s;
	BC::Matrix<float, BC::GPU> a(4,4);
	BC::Matrix<float, BC::GPU> b(4,4);
	BC::Matrix<float, BC::GPU> c(4,4);

	b.randomize(0,3);
	c.randomize(0,3);
	a = b * c;

	b.print();
	c.print();
	a.print();
//

	std::cout <<" outpusasdfasDF "<< std::endl;
//
//	a -= b * c;


	std::cout << "terminate success "<< std::endl;
}
