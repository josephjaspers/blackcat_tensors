#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"
#include "Tensor.h"

template<typename number_type>
void Tensor<number_type>::reset_post_move() {
#ifndef BLACKCAT_DISABLE_ADVANCED_CHECKS
	if (!ownership) {
		throw std::invalid_argument("ID_Tensor reset -- cannot reset a subtensor");
	}
#endif

	tensor = nullptr;
    ranks = nullptr;
	sz = 0;
    order = 0;
}

template<typename number_type>
void Tensor<number_type>::reset() {
#ifndef BLACKCAT_DISABLE_ADVANCED_CHECKS
	if (!ownership) {
		throw std::invalid_argument("ID_Tensor reset -- cannot reset a subtensor");
	}
#endif

    if (tensor)
       Tensor_Operations<number_type>::destruction(tensor);
    if (ranks)
    delete[] ranks;

    tensor = nullptr;
    ranks = nullptr;

    sz = 0;
    order = 0;
}

template<typename number_type>
void Tensor<number_type>::reshape(std::initializer_list<unsigned> new_shape) {
    unsigned new_sz = 1;
    for (auto iter = new_shape.begin(); iter != new_shape.end(); ++iter) {
        new_sz *= *iter;
    }
    if (sz != new_sz) {
        throw std::invalid_argument("reshape sz mismatch");
    }

    delete[]ranks;
    ranks = new unsigned[order];


    BC::copy(ranks, new_shape.begin(), order);
    order = new_shape.size();
}
