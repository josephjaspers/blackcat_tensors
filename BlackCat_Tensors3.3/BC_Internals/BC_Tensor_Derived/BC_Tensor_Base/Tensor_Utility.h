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
		mathlib::print(asDerived().data().getIterator(), asDerived().innerShape(), asDerived().dims(), 8);
	}
	void print(int precision) const {
		mathlib::print(asDerived().data().getIterator(), asDerived().innerShape(), asDerived().dims(), precision);
	}
	void printSparse() const {
		mathlib::printSparse(asDerived().data().getIterator(), asDerived().innerShape(), asDerived().dims(), 8);
	}
	void printSparse(int precision) const {
		mathlib::printSparse(asDerived().data().getIterator(), asDerived().innerShape(), asDerived().dims(), precision);
	}

	void write(std::ofstream& os) const {

		scalar* data = new scalar[asDerived().size()];
		mathlib::DeviceToHost(data, asDerived().data().getIterator(), asDerived().size());

		os << asDerived().dims() << ',';
		for (int i = 0; i < asDerived().dims(); ++i) {
			os << asDerived().dimension(i) << ',';
		}
		for (int i = 0; i < asDerived().size() - 1; ++i) {
			os << data[i] << ',';
		}
		os << data[asDerived().size() - 1]; //back
		os << '\n';


		delete[] data;
	}

	void read(std::ifstream& is, bool read_dimensions = true, bool overrideDimensions = true) {
		if (!is.good()) {
			std::cout << "File open error - returning " << std::endl;
			return;
		}
		std::vector<scalar> data;
		unsigned read_values = 0;

			std::string tmp;
			std::getline(is, tmp, '\n');

			std::stringstream ss(tmp);
			scalar val;

			if (ss.peek() == ',')
				ss.ignore();

			while (ss >> val) {
				data.push_back(val);
				++read_values;
				if (ss.peek() == ',')
					ss.ignore();
			}

		if (read_dimensions) {
			std::vector<int> dims((int)data[0]);
			for (int i = 0; i < dims.size(); ++i) {
				dims[i] = data[i + 1];
			}

			///THIS IS BAD DEAL WITH THIS LATER
			if (overrideDimensions) {
				switch ((int)data[0]) {
				case 0: break;//is scalar do nothing
				case 1: asDerived().resize((int)data[1]); break;
				case 2: asDerived().resize((int)data[1],(int)data[2]); break;
				case 3: asDerived().resize((int)data[1],(int)data[2],(int)data[3]); break;
				case 4: asDerived().resize((int)data[1],(int)data[2],(int)data[3],(int)data[4]); break;
				case 5: asDerived().resize((int)data[1],(int)data[2],(int)data[3],(int)data[4],(int)data[5]); break;
				default: throw std::invalid_argument("MAX DIMENSIONS READ == 5 ");
				}
			}
			mathlib::HostToDevice(asDerived().data().getIterator(), &data[data[0] + 1], asDerived().size() > data.size() ? data.size() : asDerived().size());
		} else {
			mathlib::HostToDevice(asDerived().data().getIterator(), &data[0], 			asDerived().size() > data.size() ? data.size() : asDerived().size());
		}
	}
};

}


#endif /* TENSOR_LV2_CORE_IMPL_H_ */
