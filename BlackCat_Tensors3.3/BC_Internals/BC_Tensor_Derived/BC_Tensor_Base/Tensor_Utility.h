/*
 * Tensor_Lv2_Array_impl.h
 *
 *  Created on: Jan 2, 2018
 *      Author: joseph
 */

#ifndef TENSOR_LV2_CORE_IMPL_H_
#define TENSOR_LV2_CORE_IMPL_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

namespace BC {
namespace Base {

/*
 * Defines standard utility methods related to I/O
 */

template<class deriv>
struct Tensor_Utility {

	using scalar = _scalar<deriv>;
	using mathlib = _mathlib<deriv>;

private:
	deriv& as_derived() {
		return static_cast<deriv&>(*this);
	}
	const deriv& as_derived() const {
		return static_cast<const deriv&>(*this);
	}

public:

	void print() const {
		mathlib::print(as_derived().internal().memptr(),
				as_derived().inner_shape(), as_derived().outer_shape(),
				as_derived().dims(), 8);
	}
	void print(int precision) const {
		mathlib::print(as_derived().internal().memptr(),
				as_derived().inner_shape(), as_derived().outer_shape(),
				as_derived().dims(), precision);
	}
	void printSparse() const {
		mathlib::printSparse(as_derived().internal().memptr(),
				as_derived().inner_shape(), as_derived().outer_shape(),
				as_derived().dims(), 8);
	}
	void printSparse(int precision) const {
		mathlib::printSparse(as_derived().internal().memptr(),
				as_derived().inner_shape(), as_derived().outer_shape(),
				as_derived().dims(), precision);
	}

	void write(std::ofstream& os) const {

		scalar* internal = new scalar[as_derived().size()];
		mathlib::DeviceToHost(internal, as_derived().internal().memptr(),
				as_derived().size());

		os << as_derived().dims() << ',';
		for (int i = 0; i < as_derived().dims(); ++i) {
			os << as_derived().dimension(i) << ',';
		}
		for (int i = 0; i < as_derived().size() - 1; ++i) {
			os << internal[i] << ',';
		}
		os << internal[as_derived().size() - 1]; //back
		os << '\n';

		delete[] internal;
	}

	void read_as_one_hot(std::ifstream& is, int sz = -1) {
		if (dimension_of<deriv> != 1)
			throw std::invalid_argument("one_hot only supported by vectors");

		//rescale
		if (sz > 0) {
			as_derived() = deriv(sz);
		}
		as_derived().zero(); //clear

		std::string tmp;
		std::getline(is, tmp, ',');

		as_derived()(std::stoi(tmp))  = 1;

	}

	void read(std::ifstream& is, bool read_dimensions = true,
			bool overrideDimensions = true) {
		if (!is.good()) {
			std::cout << "File open error - returning " << std::endl;
			return;
		}
		std::vector<scalar> file_data;
		scalar val;
		std::string tmp;
		unsigned read_values = 0;

		std::getline(is, tmp, '\n');

		std::stringstream ss(tmp);

		if (ss.peek() == ',' || ss.peek() == ' ' || ss.peek() == '\t')
			ss.ignore();

		while (ss >> val) {
			file_data.push_back(val);
			++read_values;
			if (ss.peek() == ',')
				ss.ignore();
		}

		if (read_dimensions) {
			std::vector<int> dims((int) file_data[0]);
			if (file_data[0] != deriv::DIMS()) {
				std::cout
						<< " attempting to read data from file of tensor of dimensions = "
						<< file_data[0]
						<< " however the reading to tensor is of dimension = "
						<< deriv::DIMS();
				throw std::invalid_argument("Invalid Tensor File");
			}
			for (int i = 0; i < dims.size(); ++i) {
				dims[i] = file_data[i + 1];
			}
			if (overrideDimensions) {

				Shape<deriv::DIMS()> shape;
				for (int i = 0; i < deriv::DIMS(); ++i) {
					shape.is()[i] = (int) file_data[i + 1];
				}

				as_derived() = deriv(shape);
			}
			mathlib::HostToDevice(as_derived().internal().memptr(),
					&file_data[file_data[0] + 1],
					as_derived().size() > file_data.size() ?
							file_data.size() : as_derived().size());
		} else {
			mathlib::HostToDevice(as_derived().internal().memptr(),
					&file_data[0],
					as_derived().size() > file_data.size() ?
							file_data.size() : as_derived().size());
		}
	}
};

}
}

#endif /* TENSOR_LV2_CORE_IMPL_H_ */
