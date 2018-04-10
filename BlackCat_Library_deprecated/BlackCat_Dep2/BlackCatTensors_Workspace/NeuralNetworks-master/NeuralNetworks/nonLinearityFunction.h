#pragma once
#ifndef nonLinearityFunct_h
#define nonLinearityFunct_h

#include "Matrices.h"
class nonLinearityFunct {
	static constexpr double e = 2.71828;
	//the current non_linearity function being used 
	int nonLinearity = 0;
public:
	void read(std::ifstream& is) {
		is >> nonLinearity;
	}
	void write(std::ofstream& os) {
		os << nonLinearity << ' ';
	}
	void setNonLinearityFunction(int i) {
		nonLinearity = i;
	}
	void setSigmoid() {
		nonLinearity = 0;
	}
	void setTanh() {
		nonLinearity = 1;
	}
private:
	//VECTOR nonlin ---VECTOR---VECTOR---VECTOR---VECTOR---VECTOR---VECTOR---VECTOR---VECTOR---VECTOR---VECTOR---VECTOR---VECTOR---VECTOR---VECTOR---VECTOR---
	//nonlinear functions & derivatives
	static Vector& sigmoid(Vector& x) {
		for (int i = 0; i < x.size(); ++i) {
			x[i] = 1 / (1 + pow(2.71828, -x[i]));
		}
		return x;
	}
	static Vector sigmoid_deriv(Vector x) {
		for (int i = 0; i < x.size(); ++i) {
			x[i] *= (1 - x[i]);
		}
		return x;
	}
	static Vector& tanh(Vector& x) {
		for (int i = 0; i < x.size(); ++i) {
			x[i] = std::tanh(x[i]);
		}
		return x;
	}
	static Vector tanh_deriv(Vector x) {
		for (int i = 0; i < x.size(); ++i) {
			x[i] = (1 - pow(x[i], 2));
		}
		return x;
	}
	static Vector& softMax(Vector& x) {
		double sum = 0;
		for (int i = 0; i < x.size(); ++i) {
			x[i] = pow(e, x[i]);
			sum += x[i];
		}
		for (int i = 0; i < x.size(); ++i) {
			x[i] /= sum;
		}
		return x;
	}
	static Vector& reLU(Vector& x) {
		for (int i = 0; i < x.size(); ++i) {
			if (x[i] < 0) {
				x[i] = 0;
			}
			else if (x[i] > 1) {
				x[i] = 1;
			}
		}
		return x;
	}
	static Vector reLU_deriv(Vector x) {
		return reLU(x);
	}

public:
	//operator for sigmoidfunction
	Vector& operator() (Vector& x) {
		switch (nonLinearity) {
		case 0: return sigmoid(x);
		case 1: return tanh(x);
		case 2: return softMax(x);
		case 3: return reLU(x);
		default: std::cout << " nonlineariy function not enabled -- returning without effect " << std::endl;
		}
	}
	//deriv and d are same methods 
	Vector deriv(const Vector& x) {
		switch (nonLinearity) {
		case 0: return sigmoid_deriv(x);
		case 1: return tanh_deriv(x);
		case 2: return sigmoid_deriv(x); //softmax deriv = sigmoid deriv
		case 3: return reLU_deriv(x);
		default: std::cout << " non linerity deriv error: set to invalid integer, returning " << std::endl;
		}
	}
	Vector d(const Vector& x) {
		switch (nonLinearity) {
		case 0: return sigmoid_deriv(x);
		case 1: return tanh_deriv(x);
		case 2: return sigmoid_deriv(x); //softmax deriv is same as sigmoid deriv
		case 3: return reLU_deriv(x);
		default: std::cout << " non linerity deriv error: set to invalid integer, returning " << std::endl;
		}
	}
	//non Lin differs as it returns a cpy of the parameter opposed to effecting it directly 
	Vector nonLin(Vector x) {
		switch (nonLinearity) {
		case 0: return sigmoid(x);
		case 1: return tanh(x);
		case 2: return softMax(x);
		case 3: return reLU(x);
		}
	}

};
#endif
