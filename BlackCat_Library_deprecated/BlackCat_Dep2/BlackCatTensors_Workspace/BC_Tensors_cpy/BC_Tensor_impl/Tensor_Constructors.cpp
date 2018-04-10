#include "Tensor.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename number_type>
Tensor<number_type>::Tensor() : ownership(true) {
	sz = 0;
	order = 0;
	tensor = nullptr;
	ranks = nullptr;
}

template<typename number_type>
Tensor<number_type>::Tensor(const Tensor<number_type>& cpy) : ownership(true) {

	sz = cpy.sz;
	order = cpy.order;

	Tensor_Operations<number_type>::initialize(tensor, sz);
	Tensor_Operations<number_type>::copy(tensor, cpy.tensor, sz);

	ranks = new unsigned[order];
	BC::copy(ranks, cpy.ranks, order);
}

template<typename number_type>
Tensor<number_type>::Tensor(Tensor<number_type> && cpy) : ownership(true) {
	order = cpy.order;
	sz = cpy.sz;

	if (cpy.ownership) {
		tensor = cpy.tensor;
		ranks = cpy.ranks;
		cpy.reset_post_move();

	} else {
		Tensor_Operations<number_type>::initialize(tensor, sz);
		Tensor_Operations<number_type>::copy(tensor, cpy.tensor, sz);

		ranks = new unsigned[order];
		BC::copy(ranks, cpy.ranks, order);
	}
}
template<typename number_type>
Tensor<number_type>::Tensor(const Tensor<number_type>& cpy, bool copy_values) :ownership(true) {
	sz = cpy.sz;
	order = cpy.order;

	Tensor_Operations<number_type>::initialize(tensor, sz);

	ranks = new unsigned[order];
	BC::copy(ranks, cpy.ranks, order);


	if (copy_values) {
			Tensor_Operations<number_type>::copy(tensor, cpy.tensor, sz);
	}
}
template<typename number_type>
Tensor<number_type>::Tensor(const Tensor<number_type>& cpy, bool copy_values, bool cpy_trans) :ownership(true) {
	sz = cpy.sz;
	order = cpy.order;

	Tensor_Operations<number_type>::initialize(tensor, sz);

	ranks = new unsigned[order];
	BC::copy(ranks, cpy.ranks, order);

	if (cpy_trans) {
		if (order > 0)
			ranks[0] = cpy.ranks[1];
		if (order > 1)
			ranks[1] = cpy.ranks[0];
	}

	if (copy_values) {
		if (cpy_trans) {
			for (int i = 0; i < totalMatrices(); ++i) {
				unsigned index = i * matrix_size();
				Tensor_Operations<number_type>::transpose(&tensor[index],
						&cpy.tensor[index], this->cols(), this->rows());
			}
		} else {
			Tensor_Operations<number_type>::copy(tensor, cpy.tensor, sz);
		}
	}
}

template<typename number_type>
Tensor<number_type>::Tensor(unsigned m, unsigned n, unsigned k, unsigned p) :
ownership(true) {

	sz = m * n * k * p;
	order = 4;

	ranks = new unsigned[order];
	Tensor_Operations<number_type>::initialize(tensor, sz);

	ranks[0] = m;
	ranks[1] = n;
	ranks[2] = k;
	ranks[3] = p;
}

template<typename number_type>
Tensor<number_type>::Tensor(unsigned m, unsigned n, unsigned k) : ownership(true) {
	sz = m * n * k;
	order = 3;

	ranks = new unsigned[order];
	Tensor_Operations<number_type>::initialize(tensor, sz);

	ranks[0] = m;
	ranks[1] = n;
	ranks[2] = k;
}

template<typename number_type>
Tensor<number_type>::Tensor(unsigned m, unsigned n) : ownership(true) {
	sz = m * n;
	order = 2;

	ranks = new unsigned[order];
	Tensor_Operations<number_type>::initialize(tensor, sz);

	ranks[0] = m;
	ranks[1] = n;
}

template<typename number_type>
Tensor<number_type>::Tensor(unsigned m) : ownership(true) {

	sz = m;
	order = 1;

	ranks = new unsigned[1];
	Tensor_Operations<number_type>::initialize(tensor, sz);

	ranks[0] = m;
}

template<typename number_type>
Tensor<number_type>::Tensor(std::initializer_list<unsigned> init_ranks) : ownership(true) {

	order = init_ranks.size();
	ranks = new unsigned[order];

	sz = 1;
	unsigned ranks_index = 0;
	for (auto iter = init_ranks.begin(); iter != init_ranks.end(); ++iter) {
		ranks[ranks_index] = *iter;
		sz *= *iter;
		++ranks_index;
	}
	Tensor_Operations<number_type>::initialize(tensor, sz);
}
template<typename number_type>
Tensor<number_type>::Tensor(const Shape& shape) : ownership(true) {

	order = shape.size();
	ranks = new unsigned[order];

	sz = 1;
	unsigned ranks_index = 0;
	for (auto iter = shape.begin(); iter != shape.end(); ++iter) {
		ranks[ranks_index] = *iter;
		sz *= *iter;
		++ranks_index;
	}
	Tensor_Operations<number_type>::initialize(tensor, sz);
}

template<typename number_type>
std::vector<unsigned> Tensor<number_type>::getShape() const {
	std::vector<unsigned> s = Shape(order);

	for (int i = 0; i < order; ++i) {
		s[i] = ranks[i];
	}
	return s;
}
template<typename number_type>
Tensor<number_type>::~Tensor() {
	if (ownership) {
		if (tensor)
			Tensor_Operations<number_type>::destruction(tensor);
		if (ranks)
			delete[] ranks;
	}
	clearSubTensorCache();
}

