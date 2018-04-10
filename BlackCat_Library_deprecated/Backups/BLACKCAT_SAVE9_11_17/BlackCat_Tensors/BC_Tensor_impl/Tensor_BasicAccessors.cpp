#include "Tensor.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename n> bool Tensor<n>::isInitialized() const {
	return tensor != nullptr;
}

template<typename n> unsigned Tensor<n>::totalMatrices() const {
	unsigned total = 1;

	for (unsigned i = 2; i < order; ++i) {
		total *= ranks[i];
	}
	return total;
}

template<typename n> unsigned Tensor<n>::matrix_size() const {
	return rows() * cols();
}
template<typename n> unsigned Tensor<n>::size() const {
	return sz;
}
template<typename n> unsigned Tensor<n>::rows() const {
	return order > 0 ? ranks[0] : 1;
}
template<typename n> unsigned Tensor<n>::cols() const {
	return order > 1 ? ranks[1] : 1;
}
template<typename n> unsigned Tensor<n>::pages() const {
	return order > 2 ? ranks[2] : 1;
}
template<typename n> bool Tensor<n>::isMatrix() const {
	return order <= 2;
}
template<typename n> bool Tensor<n>::isVector() const {
	return order <= 2 && (rows() == 1 || cols() == 1);
}
template<typename n> bool Tensor<n>::isSquare() const {
	return order < 3 ? rows() == cols() : false;
}
template<typename n> bool Tensor<n>::isScalar() const {
	return order == 0;
}
template<typename n> unsigned Tensor<n>::rank(unsigned rank_index) const {
	return rank_index > 0 && rank_index <= degree() ? ranks[rank_index - 1] : 1;
}

template<typename n> unsigned Tensor<n>::leading_dim(unsigned rank_index) const {
	return rank_index < order ? ld[rank_index] : sz;
}
template<typename n> unsigned Tensor<n>::degree() const {
	return order;
}
