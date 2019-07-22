/*
 * mnist_from_scratch.h
 *
 *  Created on: Jul 18, 2019
 *      Author: joseph
 */

#ifndef MNIST_FROM_SCRATCH_H_
#define MNIST_FROM_SCRATCH_H_

#include "BlackCat_Tensors.h"

int main() {

	using system_tag = BC::host_tag;
	using value_type = typename host_tag::default_floating_point_type;
	using mat = BC::Matrix<value_type, BC::Allocator<system_tag>>;
	using vec = BC::Vector<value_type, BC::Allocator<system_tag>>;

	/*
	 * Network Architecture:
	 * 		Dense (w * x + b) 784:256
	 *		logistic
	 *		Dense (w * x + b) 256:10
	 *		logistic
	 */

	struct dense {
		value_type lr = .003;
		mat w;
		vec b;

		dense(int inputs, int outputs):
			w(outputs, inputs),
			b(outputs) {}

		template<class Tensor>
		auto fp(const Tensor& x) {
			return w * x + b;
		}

		template<class Tensor>
		auto bp(const Tensor& dy) {
			w -= lr * dy * x.t();
			b -= lr * dy;
			return w.t() * dy;
		}
	};



	dense d1(784, 256);
	dense d2(256, 10);

	std::vector<mat> data;
	std::vector<mat> out;
	for (int i = 0; i < data.size(); ++i) {
		auto x1 = BC::logistic(d1.fp(data[i]));
		auto x2 = BC::logistic(d2.fp(x1));

		//Note: % == elementwise multiplication
		auto d2 = d2.bp(BC::logistic.dx(x2) % (x2 - out[i]));
		auto d3 = d1.bp(BC::logistic.dx(x1) % d1);
	}
}


#endif /* MNIST_FROM_SCRATCH_H_ */
