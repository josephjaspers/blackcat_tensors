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

template<class derived>
struct Tensor_Utility {

	using scalar = scalar_of<derived>;
	using mathlib = mathlib_of<derived>;

private:
	derived& as_derived() {
		return static_cast<derived&>(*this);
	}
	const derived& as_derived() const {
		return static_cast<const derived&>(*this);
	}

public:


	void print(int precision=8) const {
		BC_ARRAY_ONLY("void print(int precision=8) const");
		mathlib::print(as_derived().internal().memptr(), as_derived().inner_shape(), as_derived().outer_shape(), as_derived().dims(), precision);
	}
	void printSparse(int precision=8) const {
		BC_ARRAY_ONLY("void printSparse(int precision=8) const");
		mathlib::printSparse(as_derived().internal().memptr(), as_derived().inner_shape(), as_derived().outer_shape(), as_derived().dims(), precision);
	}

	void write(std::ofstream& os) const {
		BC_ARRAY_ONLY("write(std::ofstream& os)");

		scalar* internal = new scalar[as_derived().size()];
		mathlib::DeviceToHost(internal, as_derived().internal().memptr(), as_derived().size());

		for (int i = 0; i < as_derived().size() - 1; ++i) {
			os << internal[i] << ',';
		}
		os << internal[as_derived().size() - 1]; //back
		os << '\n';

		delete[] internal;
	}
	void write_tensor_data(std::ofstream& os) const {
		BC_ARRAY_ONLY("void write_tensor_data(std::ofstream& os)");

		scalar* internal = new scalar[as_derived().size()];
		mathlib::DeviceToHost(internal, as_derived().internal().memptr(), as_derived().size());

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
	void read_as_one_hot(std::ifstream& is) {
		BC_ARRAY_ONLY("void read_as_one_hot(std::ifstream& is)");

		if (derived::DIMS() != 1)
			throw std::invalid_argument("one_hot only supported by vectors");

//		as_derived().zero(); //clear FIXME COMPILE ISSUE WITH NVCC

		std::string tmp;
		std::getline(is, tmp, ',');

		as_derived()(std::stoi(tmp)) = 1;

	}
	void read(std::ifstream& is) {
		BC_ARRAY_ONLY("void read(std::ifstream& is)");

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
			mathlib::HostToDevice(as_derived().internal().memptr(), &file_data[0], (unsigned)as_derived().size() > file_data.size() ? file_data.size() : as_derived().size());

	}

	void read_tensor_data(std::ifstream& is, bool read_dimensions = true, bool overrideDimensions = true) {
		BC_ARRAY_ONLY("void read_tensor_data(std::ifstream& is, bool read_dimensions = true, bool overrideDimensions = true)");

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
			if (file_data[0] != derived::DIMS()) {
				std::cout << " attempting to read data from file of tensor of dimensions = " << file_data[0]
						<< " however the reading to tensor is of dimension = " << derived::DIMS();
				throw std::invalid_argument("Invalid Tensor File");
			}
			for (int i = 0; i < dims.size(); ++i) {
				dims[i] = file_data[i + 1];
			}
			if (overrideDimensions) {

				Shape<derived::DIMS()> shape;
				for (int i = 0; i < derived::DIMS(); ++i) {
					shape.is()[i] = (int) file_data[i + 1];
				}

				as_derived() = derived(shape);
			}
			mathlib::HostToDevice(as_derived().internal().memptr(), &file_data[file_data[0] + 1],
					as_derived().size() > file_data.size() ? file_data.size() : as_derived().size());
		} else {
			mathlib::HostToDevice(as_derived().internal().memptr(), &file_data[0],
					as_derived().size() > file_data.size() ? file_data.size() : as_derived().size());
		}
	}
	void read_tensor_data_as_one_hot(std::ifstream& is, int sz) {
		BC_ARRAY_ONLY("void read_tensor_data_as_one_hot(std::ifstream& is, int sz)");
		if (derived::DIMS() != 1)
			throw std::invalid_argument("one_hot only supported by vectors");

		//rescale
		if (sz > 0) {
			as_derived() = derived(sz);
		}
		//		as_derived().zero(); //clear FIXME COMPILE ISSUE WITH NVCC

		std::string tmp;
		std::getline(is, tmp, ',');

		as_derived()(std::stoi(tmp)) = 1;

	}
};

}
}

#endif /* TENSOR_LV2_CORE_IMPL_H_ */
