//
//
//#include <iostream>
//#include "BlackCat_Tensors.h"
//#include <cmath>
//
//using BC::Vector;
//using BC::Matrix;
//using BC::Scalar;
//using BC::expression;
//
//using input  = Vector<double, 2>;
//using output = Vector<double, 1>;
//
//
//
//struct data {
//
//	input i1 = {0, 0};
//	input i2 = {0, 1};
//	input i3 = {1, 1};
//	input i4 = {1, 0};
//
//	output o1 = {1};
//	output o2 = {0};
//	output o3 = {1};
//	output o4 = {0};
//};
//
//template<class U, int rows> auto g(Vector<U, rows> vec) {
//	Vector<double, rows> ret;
//	for (int i = 0; i < vec.size(); ++i) {
//		ret[i] = 1 / (1 + std::pow(2.71828, - vec[i]));
//	}
//	return ret;
//}
//
//template<class U, int rows> auto gd(Vector<U, rows> vec) {
//	Vector<double, rows> ret;
//	Scalar<double> one(1);
//	for (int i = 0; i < vec.size(); ++i) {
//		ret = (one - ret[i]) % ret[i];
//	}
//	return ret;
//}
//
//struct NN {
//
//
//
//	Matrix<double, 10, 2> w1;
//	Matrix<double, 10, 2> w1_storage;
//
//	Vector<double, 10> b1;
//
//	Matrix<double, 1, 10> w2;
//	Matrix<double, 1, 10> w2_storage;
//
//	Vector<double, 1>  b2;
//	Vector<double, 10> x2;
//
//
//	Vector<double, 1>  y;
//	Vector<double, 10> dx2;
//
//	Scalar<double> lr = Scalar<double>(.3);
//
//	void init() {
//		w1.randomize(-3 ,3);
//		w2.randomize(-3, 3);
//	}
//
//
//	int run(const input& x, const output& hyp) {
//
//
//		return 0;
//	}
//	int test(input x, output hyp) {
//
//		x2 = g(w1 * x + b1);  //fp layer 1
//		y  = g(w2 * x2 + b2); //fp layer 2
//
//		auto residual = y - hyp; 	//delta layer 2
//
//		w2_storage = residual * x2.t();				//storage for layer 1
//		w1_storage = w2.t() * residual % gd(x2);	//storage for layer 2
//
//		x.print();
//		y.print();
//
//		return 0;
//	}
//};
//
//;
//template<int in, int out>
//struct layer {
//
//	layer() {
//		w.randomize(-3, 3);
//		b.randomize(-3, 3);
//	}
//
//	Matrix<double, out, in> w;
//	Vector<double, in> x;
//	Vector<double, out> b;
//
//	Vector <double, in> dx;
//	Vector <double, out> dy;
//	Matrix <double, out, in> w_storage;
//
////	template<class U>
//	auto fp(const Vector<double, in> input) {
//		x = input;
//		return g(w * x  + b);
//	}
//
////	template<class U>
//	auto bp(const Vector<double, out> delta) {
//		w -= delta * x.t();
//		return w.t() * delta ;//% gd(x);
//	}
//
//};
//
//
//
//
//int main() {
//	data set;
//
//	layer<2, 10> l1;
//	layer<10, 1> l2;
//
//	for (int i = 0; i < 4; ++i) {
//	std::cout << " fp" << std::endl;
//	auto x1 = l1.fp(set.i1);
//	x1.print();
//	auto y  = l2.fp(x1);
//	y.print();
//	std::cout << "res" << std::endl;
//	auto error = set.o1 - y;
//	error.print();
//	std::cout << "bp" << std::endl;
//	auto dy    = l2.bp(error);
//
//	}
//
//
//
//
//
//}
//
