/*
 * NonLinearityFunction.h
 *
 *  Created on: Aug 10, 2017
 *      Author: joseph
 */

#include "BC_NN_Functions.h"

#ifndef NONLINEARITYFUNCTION_H_
#define NONLINEARITYFUNCTION_H_

//controls sigmoid/tanh -- will be update
struct nonLinearityFunction {
	///nonLinearity nonLin = sigmoid;
public:
	tensor operator()(tensor x) {
		nonLin::sigmoid(x);
		return x;
	}
	tensor d(tensor x) {
		nonLin::sigmoid_deriv(x);
		return x;
	}
};

//bpStorages - controls back propagation data
#include <unordered_map>
#include <mutex>
struct bpStorage {
	std::vector<tensor> bp_storages;

	void store(tensor data) {
		bp_storages.push_back(data);
	}
	tensor& last() {
		return bp_storages.back();
	}
	tensor poll_last() {
		tensor data = std::move(bp_storages.back());
		bp_storages.pop_back();

		return std::move(data);
	}
	void clear() {
		bp_storages.clear();
	}
	void clearAll() {
		bp_storages.clear();
	}

	const tensor& operator()() {
		return bp_storages.back();
	}
	int size() {
		return bp_storages.size();
	}
};

//gradient storages -- updates weights/gradients -- assists with multithreading

#endif /* NONLINEARITYFUNCTION_H_ */
