#include "Tensor.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename number_type>
Tensor<number_type>::Tensor(const Tensor<number_type>& m1, const Tensor<number_type>& m2) :
tensor_ownership(true), rank_ownership(true), ld_ownership(true), subTensor(false) {
	//DOTPRODUCT CONSTRUCTOR
	order = m2.order;
	ranks = new unsigned[order];
	BC::copy(ranks, m2.ranks, order);



	ranks[0] = m1.rows();
	if (order > 1)
	ranks[1] = m2.cols();

	ld = new unsigned[order];
	BC::init_leading_dimensions(ld, ranks, order);
	sz = BC::calc_sz(ranks, order);

	Tensor_Operations<number_type>::initialize(tensor, size());
}

