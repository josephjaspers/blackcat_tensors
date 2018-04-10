#include "Tensor.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename number_type> void Tensor<number_type>::printDimensions() const {
    for (int i = 0; i < order; ++i) {
        std::cout << "[" << ranks[i] << "]";
    }
}

template<typename n>
bool Tensor<n>::same_dimensions(const Tensor<n>& t) const {
	if (degree() != t.degree()) {
		return false;
	}
	for (unsigned i = 0; i < order; ++i) {
		if (rank(i) != t.rank(i)) {
			return false;
		}
	}
	return true;
}

template<typename n>
void Tensor<n>::assert_same_dimensions(const Tensor<n>& t) const {
#ifndef BLACKCAT_DISABLE_RUNTIME_CHECKS
	if (!same_dimensions(t)) {
		throw std::invalid_argument("same_dimensions assert FAIL");
	}
#endif
}

template<typename n>
bool Tensor<n>::dotProduct_dimensions(const Tensor<n>& t) const {
	return this->cols() == t.rows();
}

template<typename n>
void Tensor<n>::assert_dotProduct_dimensions(const Tensor<n>& t) const {
#ifndef BLACKCAT_DISABLE_RUNTIME_CHECKS
	if (!dotProduct_dimensions(t)) {
		std::cout << " this->dimensions "; printDimensions(); std::cout<<std::endl;
		std::cout << " param dimensions "; t.printDimensions(); std::cout<<std::endl;

		throw std::invalid_argument("dotproduct_dim assert FAIL");
	}
#endif
}


template<typename n>
void Tensor<n>::assert_isVector(const Tensor<n>& t) {
#ifndef BLACKCAT_DISABLE_RUNTIME_CHECKS
	if (!t.isVector()) {
		std::cout << " param dimensions "; t.printDimensions(); std::cout<<std::endl;

		throw std::invalid_argument("isVector assert FAIL");
	}
#endif
}

