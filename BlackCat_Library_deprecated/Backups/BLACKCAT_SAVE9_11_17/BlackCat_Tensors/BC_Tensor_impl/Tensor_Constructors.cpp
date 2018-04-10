#include "Tensor.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename number_type>
Tensor<number_type>::Tensor() : tensor_ownership(true), rank_ownership(true), ld_ownership(true), subTensor(false) {
	sz = 0;
	order = 0;
	tensor = nullptr;
	ranks = nullptr;
	ld = nullptr;
}

template<typename number_type>
Tensor<number_type>::Tensor(const Tensor<number_type>& cpy) : tensor_ownership(true), rank_ownership(true), ld_ownership(true),subTensor(false) {
	order = cpy.order;
	sz = cpy.sz;

	ranks = new unsigned[order];
	BC::copy(ranks, cpy.ranks, order);

	ld = new unsigned[order];
	BC::init_leading_dimensions(ld, cpy.ranks, order);

	Tensor_Operations<number_type>::initialize(tensor, size());

	cpy.subTensor ?
			Tensor_Operations<number_type>::copy(tensor, ranks, order, ld, cpy.tensor, cpy.ld)
			:
			Tensor_Operations<number_type>::copy(tensor, cpy.tensor, size());
}

template<typename number_type>
Tensor<number_type>::Tensor(Tensor<number_type> && cpy) : tensor_ownership(true), rank_ownership(true), ld_ownership(true), subTensor(false) {

	order = cpy.order;
	sz = cpy.sz;


	if (cpy.tensor_ownership) {
		tensor = cpy.tensor;
		ranks = cpy.ranks;
		ld = cpy.ld;
		cpy.reset_post_move();

	} else {
		Tensor_Operations<number_type>::initialize(tensor, size());

		ranks = new unsigned[order];
		BC::copy(ranks, cpy.ranks, order);

		ld = new unsigned[order];
		BC::init_leading_dimensions(ld, cpy.ranks, order);

		cpy.subTensor ?
				Tensor_Operations<number_type>::copy(tensor, ranks, order, ld, cpy.tensor, cpy.ld)
				:
				Tensor_Operations<number_type>::copy(tensor, cpy.tensor, size());
	}
}
template<typename number_type>
Tensor<number_type>::Tensor(const Tensor<number_type>& cpy, bool copy_values) :tensor_ownership(true), rank_ownership(true), ld_ownership(true), subTensor(false) {
	sz = cpy.sz;
	order = cpy.order;

	ranks = new unsigned[order];
	BC::copy(ranks, cpy.ranks, order);

	ld = new unsigned[order];
	BC::init_leading_dimensions(ld, cpy.ranks, order);

	Tensor_Operations<number_type>::initialize(tensor, size());
	if (copy_values) {
		cpy.subTensor ?
				Tensor_Operations<number_type>::copy(tensor, ranks, order, ld, cpy.tensor, cpy.ld)
				:
				Tensor_Operations<number_type>::copy(tensor, cpy.tensor, size());
	}
}
template<typename number_type>
Tensor<number_type>::Tensor(const Tensor<number_type>& cpy, bool copy_values, bool cpy_trans) : tensor_ownership(true), rank_ownership(true), ld_ownership(true), subTensor(false) {
	sz = cpy.sz;

	order = cpy.order == 1 ? 2 : cpy.order;
	//order = cpy.order;



	ranks = new unsigned[order];
	ld = new unsigned[order];

	BC::copy(ranks, cpy.ranks, order);

	if (cpy_trans) {
		if (order > 0)
			ranks[0] = cpy.cols();
		if (order > 1)
			ranks[1] = cpy.rows();
	}
	BC::init_leading_dimensions(ld, ranks, order);
	Tensor_Operations<number_type>::initialize(tensor, sz);

	if (copy_values) {
		if (cpy_trans) {
			for (int i = 0; i < totalMatrices(); ++i) {
				unsigned index = i * matrix_size();

				Tensor_Operations<number_type>::transpose(&tensor[index], ld[1], &cpy.tensor[index], cpy.rows(), cpy.cols(), cpy.ld[1]);
			}
		}
		 else {
			cpy.subTensor ?
							Tensor_Operations<number_type>::copy(tensor, ranks, order, ld, cpy.tensor, ld)
							:
							Tensor_Operations<number_type>::copy(tensor, cpy.tensor, size());
		}
	}
}

template<typename number_type>
Tensor<number_type>::Tensor(unsigned m, unsigned n, unsigned k, unsigned p) :
tensor_ownership(true), rank_ownership(true), ld_ownership(true), subTensor(false) {

	order = 4;
	ranks = new unsigned[order];

	ranks[0] = m;
	ranks[1] = n;
	ranks[2] = k;
	ranks[3] = p;

	sz = BC::calc_sz(ranks, order);

	ld = new unsigned[order];
	BC::init_leading_dimensions(ld, ranks, order);

	Tensor_Operations<number_type>::initialize(tensor, size());

}

template<typename number_type>
Tensor<number_type>::Tensor(unsigned m, unsigned n, unsigned k) :
tensor_ownership(true), rank_ownership(true), ld_ownership(true), subTensor(false) {
	order = 3;

	ranks = new unsigned[order];

	ranks[0] = m;
	ranks[1] = n;
	ranks[2] = k;

	ld = new unsigned[order];
	BC::init_leading_dimensions(ld, ranks, order);
	sz = m * n * k;
	Tensor_Operations<number_type>::initialize(tensor, size());
}

template<typename number_type>
Tensor<number_type>::Tensor(unsigned m, unsigned n) :
tensor_ownership(true), rank_ownership(true), ld_ownership(true), subTensor(false){
	order = 2;

	ranks = new unsigned[order];

	ranks[0] = m;
	ranks[1] = n;

	sz = m * n;

	ld = new unsigned[order];
	BC::init_leading_dimensions(ld, ranks, order);
	Tensor_Operations<number_type>::initialize(tensor, size());
}

template<typename number_type>
Tensor<number_type>::Tensor(unsigned m) :
tensor_ownership(true), rank_ownership(true), ld_ownership(true), subTensor(false) {
	order = 1;

	ranks = new unsigned[order];

	ranks[0] = m;
	sz = m;
	ld = new unsigned[order];
		BC::init_leading_dimensions(ld, ranks, order);
	Tensor_Operations<number_type>::initialize(tensor, size());
}

template<typename number_type>
Tensor<number_type>::Tensor(std::initializer_list<unsigned> init_ranks) :
tensor_ownership(true), rank_ownership(true), ld_ownership(true), subTensor(false) {

	order = init_ranks.size();
	ranks = new unsigned[order];

	unsigned ranks_index = 0;
	sz = 1;
	for (auto iter = init_ranks.begin(); iter != init_ranks.end(); ++iter) {
		sz *= *iter;
		ranks[ranks_index] = *iter;
		++ranks_index;
	}

	ld = new unsigned[order];
	BC::init_leading_dimensions(ld, ranks, order);

	Tensor_Operations<number_type>::initialize(tensor, size());
}
typedef std::vector<unsigned> Shape;

template<typename number_type>
Tensor<number_type>::Tensor(const Shape& shape) :
tensor_ownership(true), rank_ownership(true), ld_ownership(true), subTensor(false) {

	order = shape.size();
	ranks = new unsigned[order];

	sz = 1;
	BC::copy(ranks, &shape[0], shape.size());
	for (unsigned i = 0; i < shape.size(); ++i) {
		sz *= shape[i];
	}
	ld = new unsigned[order];
	BC::init_leading_dimensions(ld, ranks, order);
	Tensor_Operations<number_type>::initialize(tensor, size());

//	printDimensions();
//	print();
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
	//std::cout << " destructor " << std::endl;
	if (tensor_ownership) {
//		std::cout << " deleting tensor "<< std::endl;
		delete[] tensor;
//		std::cout << "success " << std::endl;
	}
	if (rank_ownership) {
//		std::cout << "deleting ranks" << std::endl;

		delete[] ranks;
//		std::cout << "success " << std::endl;

	}
	if (ld_ownership) {
//		std::cout << "deleting ld" << std::endl;

		delete[] ld;
//		std::cout << "success " << std::endl;

	}
	//if (self_transposed) {
		//std::cout << "deleting self t" << std::endl;

		//delete self_transposed;
		//std::cout << "success " << std::endl;

	//}
//	std::cout << " clear sub " << std::endl;
	clearTensorCache();
	//std::cout << " destructor finished " << std::endl;
}

