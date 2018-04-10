#include "Vector.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename number_type>
bool Vector<number_type>::same_size(const Tensor<number_type>& t) const {
	return this->size() == t.size();
}
template<typename number_type>
void Vector<number_type>::assert_same_size(const Tensor<number_type>& t) const {
#ifndef BLACKCAT_DISABLE_RUNTIME_CHECKS
	if (!same_size(t)) {
		throw std::invalid_argument("same size mismatch");
	}
#endif
}
