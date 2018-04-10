#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"
#include "Tensor.h"

template<typename number_type>
void Tensor<number_type>::reset_post_move() {
#ifndef BLACKCAT_DISABLE_ADVANCED_CHECKS
	if (!tensor_ownership) {
		throw std::invalid_argument("ID_Tensor reset -- cannot reset a subtensor");
	}
#endif
	sz = 0;
	tensor = nullptr;
    ranks = nullptr;
    ld = nullptr;
   // delete self_transposed;
    //self_transposed = nullptr;
	sz = 0;
    order = 0;

    //????may not eed this
    clearTensorCache();
    alertUpdate();
}

template<typename number_type>
void Tensor<number_type>::reset() {
#ifndef BLACKCAT_DISABLE_ADVANCED_CHECKS
	if (!tensor_ownership) {
		throw std::invalid_argument("ID_Tensor reset -- cannot reset a subtensor");
	}
#endif

   Tensor_Operations<number_type>::destruction(tensor);
   delete[] ranks;
   delete[] ld;
 //  delete self_transposed;

    sz = 0;
    tensor = nullptr;
    ranks = nullptr;
    ld = nullptr;
  //  self_transposed = nullptr;
    order = 0;

    clearTensorCache();
    alertUpdate();
}

template<typename number_type>
Tensor<number_type>& Tensor<number_type>::reshape(std::initializer_list<unsigned> new_shape) {
#ifndef BLACKCAT_DISABLE_ADVANCED_CHECKS
	if (!tensor_ownership) {
		throw std::invalid_argument("ID_Tensor reset -- cannot reset a subtensor");
	}
	unsigned new_sz = 1;
    for (auto iter = new_shape.begin(); iter != new_shape.end(); ++iter) {
        new_sz *= *iter;
    }

    if (size() != new_sz) {
        throw std::invalid_argument("reshape sz mismatch");
    }
    #endif


    delete[]ld;
    delete[]ranks;
    order = new_shape.size();

    ranks = new unsigned[order];
    ld = new unsigned[order];


    BC::copy(ranks, new_shape.begin(), order);
	BC::init_leading_dimensions(ld, ranks, order);

    clearTensorCache();
    alertUpdate();
    return * this;
}
template<typename number_type>
Tensor<number_type>& Tensor<number_type>::flatten() {
#ifndef BLACKCAT_DISABLE_ADVANCED_CHECKS
	if (!tensor_ownership) {
		throw std::invalid_argument("ID_Tensor reset -- cannot reset a subtensor");
	}
#endif
	reshape({size(), 1});
	order = 1;
    alertUpdate();
    return * this;
}

