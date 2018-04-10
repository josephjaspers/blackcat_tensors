#include "Tensor.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"
//
template<typename number_type> number_type& Tensor<number_type>::operator()(unsigned index) {
   return tensor[index];
}

template<typename number_type> inline const number_type&  Tensor<number_type>::operator()(unsigned index) const {
	   return tensor[index];
}

template<typename number_type>
Tensor<number_type>::Tensor(const Tensor<number_type>& super_tensor, unsigned index) : ownership(false) {
	unsigned tensor_index = 1;
	for (unsigned i = 0; i < super_tensor.order - 1; ++i) {
		tensor_index *= super_tensor.ranks[i];
	}
	tensor_index *= index;

	this->tensor = &super_tensor.tensor[tensor_index];
	this->ranks = super_tensor.ranks;
	this->order = super_tensor.order - 1;
	this->sz = super_tensor.sz / this->ranks[super_tensor.order - 1];
}


template<typename number_type> Tensor<number_type>* Tensor<number_type>::generateSub(unsigned index) const {
	return new Tensor<number_type>(*this, index);
}

template<typename number_type> Tensor<number_type>& Tensor<number_type>::operator[](unsigned index) {

#ifndef BLACKCAT_DISABLE_RUNTIME_CHECKS
	if (index >= ranks[order - 1]) {
	throw std::invalid_argument("Operator[] out of bounds -- Order: " + std::to_string(order) + " Index: " + std::to_string(index));
}
#endif

	if (IndexTensorMap.find(index) == IndexTensorMap.end()) {
		IndexTensorMap[index] = generateSub(index);
		return *(IndexTensorMap[index]);
	} else {
		return *(IndexTensorMap[index]);
	}
}

template<typename number_type>const Tensor<number_type>& Tensor<number_type>::operator[](unsigned index) const {

#ifndef BLACKCAT_DISABLE_RUNTIME_CHECKS
	if (index >= ranks[order - 1]) {
	throw std::invalid_argument("Operator[] out of bounds -- Order: " + std::to_string(order) + " Index: " + std::to_string(index));
}
#endif

	if (IndexTensorMap.find(index) == IndexTensorMap.end()) {
		IndexTensorMap[index] = generateSub(index);
		return *(IndexTensorMap[index]);
	} else {
		return *(IndexTensorMap[index]);
	}
}


template<typename number_type>
void Tensor<number_type>::clearSubTensorCache() const {
	if(IndexTensorMap.size() > 0) {
		for (auto it = IndexTensorMap.begin(); it != IndexTensorMap.end(); ++it) {
			delete it->second;
		}
	}
}
