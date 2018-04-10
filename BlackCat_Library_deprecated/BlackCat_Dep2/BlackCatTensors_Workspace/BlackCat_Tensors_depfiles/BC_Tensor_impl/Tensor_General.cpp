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

	if (sz > 200) {
		std::cout << " print limit exceeded - skipping - " << std::endl;
		printDimensions();
		return;
	}

    std::cout << "-" << std::endl;
    for (int i = 0; i < totalMatrices(); ++i) {
        std::cout << "[";
        for (int j = 0; j < matrix_size(); ++j) {
            if (j != 0)
                if (j % this->cols() == 0)
                    std::cout << std::endl << " ";

            std::cout << " " << tensor[j + i * matrix_size()];
        }
        std::cout << " ]" << std::endl;
    }
    std::cout << "-" << std::endl;
}

template<typename number_type> void Tensor<number_type>::printDimensions() const {
	std::cout << "[" << rows() << "]";
	std::cout << "[" << cols() << "]";
    for (int i = 2; i < order; ++i) {
        std::cout << "[" << ranks[i] << "]";
    }
}
