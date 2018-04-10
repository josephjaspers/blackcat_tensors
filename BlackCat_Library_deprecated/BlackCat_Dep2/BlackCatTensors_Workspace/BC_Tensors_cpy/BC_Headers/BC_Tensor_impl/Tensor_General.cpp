#include "Tensor.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename number_type>
inline void Tensor<number_type>::fill(number_type val) {
    Tensor_Operations<number_type>::fill(tensor, val, sz);
}

template<typename number_type>
inline void Tensor<number_type>::randomize(number_type lower_bound, number_type upper_bound) {
    Tensor_Operations<number_type>::randomize(tensor, lower_bound, upper_bound, sz);
}

template<typename number_type>
inline Tensor<number_type> Tensor<number_type>::transpose() const {
    Tensor trans(*this, true, true);
    return trans;
}


template<typename number_type> void Tensor<number_type>::print() const {

	if (degree() > 2) {
		std::cout << "---" <<degree() << "---"<<std::endl;
		for (int i = 0; i < rank(degree()); ++i) {
			(this->operator[](i)).print();
		}
	} else {		std::cout << "[ ";
		for (unsigned r = 0; r < rows(); ++r) {
			for (unsigned c = 0; c< cols(); ++c) {
				std::cout << tensor[c * rows() + r] << " ";
			}
			if (r != rows() - 1)
			std::cout << std::endl << "  ";
		}
		std::cout << "]" << std::endl;
	}
}

template<typename number_type> void Tensor<number_type>::printDimensions() const {
	std::cout << "[" << rows() << "]";
	std::cout << "[" << cols() << "]";
    for (int i = 2; i < order; ++i) {
        std::cout << "[" << ranks[i] << "]";
    }
}
