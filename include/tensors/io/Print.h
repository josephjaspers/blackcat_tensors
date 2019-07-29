/*
 * Print.h
 *
 *  Created on: Jul 29, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_TENSORS_TENSORS_UTILITY_PRINT_H_
#define BLACKCAT_TENSORS_TENSORS_UTILITY_PRINT_H_

#include <string>
//#include <BlackCat_TypeTraits.h>

namespace BC {
namespace tensors {
namespace io {

//TODO
template<class ValueType>
static std::string format_value(const ValueType& s, BC::size_t precision, bool sparse=false) {
	std::string fstr  = !sparse || std::abs(s) > .1 ? std::to_string(s) : "";
	if (fstr.length() < (unsigned)precision)
		return fstr.append(precision - fstr.length(), ' ');
	else
		return fstr.substr(0, precision);
}


template<class Tensor>
std::string to_string(const Tensor& tensor, int precision, bool sparse, BC::traits::Integer<2>) {
	std::string s = "";
	for (BC::size_t m = 0; m < tensor.rows(); ++m) {
		s += "[";
		for (BC::size_t n = 0; n < tensor.cols(); ++n) {
			s += format_value(tensor[n][m].memptr()[0], precision, sparse) + ", ";
		}
		s += "]\n";
	}
	return s;
}


template<class Tensor>
std::string to_string(const Tensor& tensor, int precision, int sparse, BC::traits::Integer<1>) {
	std::string s = "[";
	for (BC::size_t m = 0; m < tensor.rows(); ++m) {
		s += format_value(tensor[m].memptr()[0], precision, sparse) +", ";
	}
	s += "]\n";
	return s;
}

template<class Tensor>
std::string to_string(const Tensor& tensor, int precision, int sparse, BC::traits::Integer<0>) {
	return "[" + format_value(tensor.memptr()[0], precision, sparse) + "]\n";
}

template<class Tensor, int X>
std::string to_string(const Tensor& tensor, int precision, int sparse, BC::traits::Integer<X>) {
	std::string s = "";
	for (auto block : tensor) {
		s +=  "[" + to_string(block, precision, sparse, BC::traits::Integer<X-1>()) + "]\n";
	}
	return s;
}


}
}
}

#endif /* PRINT_H_ */
