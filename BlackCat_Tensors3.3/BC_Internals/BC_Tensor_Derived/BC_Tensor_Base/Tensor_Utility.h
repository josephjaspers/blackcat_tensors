/*
 * Tensor_Lv2_Core_impl.h
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
		mathlib::print(as_derived().internal().memptr(), as_derived().inner_shape(), as_derived().outer_shape(), as_derived().dims(), 8);
	}
	void print(int precision) const {
		mathlib::print(as_derived().internal().memptr(), as_derived().inner_shape(), as_derived().outer_shape(), as_derived().dims(), precision);
	}
	void printSparse() const {
		mathlib::printSparse(as_derived().internal().memptr(), as_derived().inner_shape(), as_derived().outer_shape(), as_derived().dims(), 8);
	}
	void printSparse(int precision) const {
		mathlib::printSparse(as_derived().internal().memptr(), as_derived().inner_shape(), as_derived().outer_shape(), as_derived().dims(), precision);
	}

	void write(std::ofstream& os) const {

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

	void read(std::ifstream& is, bool read_dimensions = true, bool overrideDimensions = true) {
		if (!is.good()) {
			std::cout << "File open error - returning " << std::endl;
			return;
		}
		std::vector<scalar> internal;
		unsigned read_values = 0;

			std::string tmp;
			std::getline(is, tmp, '\n');

			std::stringstream ss(tmp);
			scalar val;

			if (ss.peek() == ',')
				ss.ignore();

			while (ss >> val) {
				internal.push_back(val);
				++read_values;
				if (ss.peek() == ',')
					ss.ignore();
			}

		if (read_dimensions) {
			std::vector<int> dims((int)internal[0]);
			for (int i = 0; i < dims.size(); ++i) {
				dims[i] = internal[i + 1];
			}

			///THIS IS BAD DEAL WITH THIS LATER
			if (overrideDimensions) {
				switch ((int)internal[0]) {
				case 0: break;//is scalar do nothing
				case 1: as_derived().resize((int)internal[1]); break;
				case 2: as_derived().resize((int)internal[1],(int)internal[2]); break;
				case 3: as_derived().resize((int)internal[1],(int)internal[2],(int)internal[3]); break;
				case 4: as_derived().resize((int)internal[1],(int)internal[2],(int)internal[3],(int)internal[4]); break;
				case 5: as_derived().resize((int)internal[1],(int)internal[2],(int)internal[3],(int)internal[4],(int)internal[5]); break;
				default: throw std::invalid_argument("MAX DIMENSIONS READ == 5 ");
				}
			}
			mathlib::HostToDevice(as_derived().internal().memptr(), &internal[internal[0] + 1], as_derived().size() > internal.size() ? internal.size() : as_derived().size());
		} else {
			mathlib::HostToDevice(as_derived().internal().memptr(), &internal[0], 			as_derived().size() > internal.size() ? internal.size() : as_derived().size());
		}
	}
};
}
}


#endif /* TENSOR_LV2_CORE_IMPL_H_ */
