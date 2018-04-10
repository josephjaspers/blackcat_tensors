#include "Tensor.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename number_type>
Tensor<number_type>::Tensor(const Tensor<number_type>& m1, const Tensor<number_type>& m2) : ownership(true) {
	//DOTPRODUCT CONSTRUCTOR
	order = m2.order;
	ranks = new unsigned[order];

	BC::copy(ranks, m2.ranks, order);

	ranks[0] = m1.rows();

	if (order > 1)
	ranks[1] = m2.cols();

	sz = (m2.size() / m2.rows()) * m1.rows();
	Tensor_Operations<number_type>::initialize(tensor, sz);
}

