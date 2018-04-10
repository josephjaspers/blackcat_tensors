#include "Tensor.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename number_type, class TensorOperations>
bool Tensor<number_type, TensorOperations>::same_size(const Tensor<number_type, TensorOperations>& t) const {
	return this->size() == t.size();
}
template<typename number_type, class TensorOperations>
void Tensor<number_type, TensorOperations>::assert_same_size(const Tensor<number_type, TensorOperations>& t) const {
#ifndef BLACKCAT_DISABLE_RUNTIME_CHECKS
	if (!same_size(t)) {
		throw std::invalid_argument("same size mismatch");
	}
#endif
}
