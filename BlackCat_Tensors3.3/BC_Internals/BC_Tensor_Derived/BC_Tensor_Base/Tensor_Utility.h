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
	deriv& asDerived() {
		return static_cast<deriv&>(*this);
	}
	const deriv& asDerived() const {
		return static_cast<const deriv&>(*this);
	}

public:

	void print() const {
		mathlib::print(asDerived().internal().getIterator(), asDerived().inner_shape(), asDerived().outer_shape(), asDerived().dims(), 8);
	}
	void print(int precision) const {
		mathlib::print(asDerived().internal().getIterator(), asDerived().inner_shape(), asDerived().outer_shape(), asDerived().dims(), precision);
	}
	void printSparse() const {
		mathlib::printSparse(asDerived().internal().getIterator(), asDerived().inner_shape(), asDerived().outer_shape(), asDerived().dims(), 8);
	}
	void printSparse(int precision) const {
		mathlib::printSparse(asDerived().internal().getIterator(), asDerived().inner_shape(), asDerived().outer_shape(), asDerived().dims(), precision);
	}

	void write(std::ofstream& os) const {

		scalar* internal = new scalar[asDerived().size()];
		mathlib::DeviceToHost(internal, asDerived().internal().getIterator(), asDerived().size());

		os << asDerived().dims() << ',';
		for (int i = 0; i < asDerived().dims(); ++i) {
			os << asDerived().dimension(i) << ',';
		}
		for (int i = 0; i < asDerived().size() - 1; ++i) {
			os << internal[i] << ',';
		}
		os << internal[asDerived().size() - 1]; //back
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
				case 1: asDerived().resize((int)internal[1]); break;
				case 2: asDerived().resize((int)internal[1],(int)internal[2]); break;
				case 3: asDerived().resize((int)internal[1],(int)internal[2],(int)internal[3]); break;
				case 4: asDerived().resize((int)internal[1],(int)internal[2],(int)internal[3],(int)internal[4]); break;
				case 5: asDerived().resize((int)internal[1],(int)internal[2],(int)internal[3],(int)internal[4],(int)internal[5]); break;
				default: throw std::invalid_argument("MAX DIMENSIONS READ == 5 ");
				}
			}
			mathlib::HostToDevice(asDerived().internal().getIterator(), &internal[internal[0] + 1], asDerived().size() > internal.size() ? internal.size() : asDerived().size());
		} else {
			mathlib::HostToDevice(asDerived().internal().getIterator(), &internal[0], 			asDerived().size() > internal.size() ? internal.size() : asDerived().size());
		}
	}
};
}
}


#endif /* TENSOR_LV2_CORE_IMPL_H_ */
