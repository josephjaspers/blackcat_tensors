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

struct features {

	features(std::size_t precision_,
			bool pretty_=true,
			bool sparse_=true):
		precision(precision_),
		pretty(pretty_),
		sparse(sparse_) {}


	std::size_t precision = 5;
	bool pretty = false;
	bool sparse = false;
};

//TODO
template<class ValueType>
static std::string format_value(const ValueType& s,  features f) {
	std::string fstr  = !f.sparse || std::abs(s) > .1 ? std::to_string(s) : "";
	if (fstr.length() < (unsigned)f.precision)
		return fstr.append(f.precision - fstr.length(), ' ');
	else {
		std::string substr = fstr.substr(0, f.precision);
		if (std::find(substr.begin(), substr.end(), '.') != substr.end()) {
			return fstr.substr(0, f.precision);
		} else {
			return fstr;
		}
	}
}


template<class Tensor>
std::string to_string(const Tensor& tensor, features f, BC::traits::Integer<2>) {
	std::string s = "";
	if (f.pretty)
		s += "[";

	for (BC::size_t m = 0; m < tensor.rows(); ++m) {

		if (f.pretty)
			s += "[";

		for (BC::size_t n = 0; n < tensor.cols(); ++n) {
			s += format_value(tensor[n][m].data()[0], f);

			if (n != tensor.cols() - 1)
				s+=", ";
		}

		if (f.pretty)
			s += "]";
		if (m != tensor.rows()-1)
		s += '\n';
	}

	if (f.pretty)
		s += "]";
	return s;
}


template<class Tensor>
std::string to_string(const Tensor& tensor, features f, BC::traits::Integer<1>) {
	std::string s;

	if (f.pretty)
		s += "[";

	for (BC::size_t m = 0; m < tensor.rows(); ++m) {
		s += format_value(tensor[m].data()[0], f);

		if(m!=tensor.rows()-1)
			s+=", ";
	}

	if (f.pretty)
		s += ']';
	return s;
}

template<class Tensor>
std::string to_string(const Tensor& tensor,  features f, BC::traits::Integer<0>) {
	std::string s;

	if (f.pretty)
		s += '[';

	s += format_value(tensor.data()[0], f);

	if (f.pretty)
		s += "]";

	return s;
}

template<class Tensor, int X>
std::string to_string(const Tensor& tensor, features f, BC::traits::Integer<X>) {
	std::string s;

	if (f.pretty)
		s += '[';

	for (auto it = tensor.nd_begin(); it != tensor.nd_end(); ++it) {
		s += to_string(*it, f, BC::traits::Integer<X-1>());

		if (it != tensor.nd_end() - 1)
			s += '\n';
	}

	if (f.pretty)
		s += ']';
	return s;
}


}
}
}

#endif /* PRINT_H_ */
