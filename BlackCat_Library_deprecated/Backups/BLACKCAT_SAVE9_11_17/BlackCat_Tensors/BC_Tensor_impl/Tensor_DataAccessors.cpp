#include "Tensor.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"


//------------------------------------- Scalar Accessor-------------------------------------//

//template<typename number_type> Scalar<number_type> Tensor<number_type>::operator()(unsigned index) {
//   return ID_Scalar<number_type, Tensor<number_type>>(*this, &tensor[index]);
//}
//
//template<typename number_type> inline const Scalar<number_type>  Tensor<number_type>::operator()(unsigned index) const {
//	   return ID_Scalar<number_type, Tensor<number_type>>(*this, &tensor[index]);
//}

template<typename number_type> number_type& Tensor<number_type>::operator()(unsigned index) {
	alertUpdate();
	return tensor[index];
}

template<typename number_type> const number_type& Tensor<number_type>::operator()(unsigned index) const {
	alertUpdate();
	return tensor[index];
}

//------------------------------------- Index Tensor Constructor------------------------------------//


template<typename number_type>
Tensor<number_type>::Tensor(const Tensor<number_type>& super_tensor, unsigned index) : tensor_ownership(false), subTensor(super_tensor.subTensor),
rank_ownership(false), ld_ownership(false) {
	//unsigned tensor_index = super_tensor.leading_dimensions[super_tensor.order - 1] * index;


	unsigned tensor_index = 1;
	for (unsigned i = 0; i < super_tensor.order - 1; ++i) {
		tensor_index *= super_tensor.ranks[i];
	}
	tensor_index *= index;

	this->parent = &super_tensor;
	this->tensor = &super_tensor.tensor[tensor_index];
	this->ranks = super_tensor.ranks;
	this->ld = super_tensor.ld;
	this->order = super_tensor.order - 1;

	this->sz = super_tensor.sz / super_tensor.ranks[order];
}

//------------------------------------- Generate Index Tensor-------------------------------------//

template<typename number_type> Tensor<number_type>* Tensor<number_type>::generateIndexTensor(unsigned index) const {
	return new Tensor<number_type>(*this, index);
}

//------------------------------------- Index Tensor Operator-------------------------------------//


template<typename number_type> Tensor<number_type>& Tensor<number_type>::operator[](unsigned index) {

#ifndef BLACKCAT_DISABLE_RUNTIME_CHECKS
	if (index >= ranks[order - 1]) {
	throw std::invalid_argument("Operator[] out of bounds -- Order: " + std::to_string(order) + " Index: " + std::to_string(index));
}
#endif

	if (IndexTensorMap.find(index) == IndexTensorMap.end()) {
		IndexTensorMap[index] = generateIndexTensor(index);
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
		IndexTensorMap[index] = generateIndexTensor(index);
		return *(IndexTensorMap[index]);
	} else {
		return *(IndexTensorMap[index]);
	}
}


template<typename number_type>
void Tensor<number_type>::clearTensorCache() {
	if(IndexTensorMap.size() > 0) {
		//std::cout << "deleting index tensor" << std::endl;
		for (auto it = IndexTensorMap.begin(); it != IndexTensorMap.end(); ++it) {
			delete it->second;
		}
	}
	if(SubTensorMap.size() > 0) {
//		std::cout << SubTensorMap.size() << std::endl;
//		std::cout << " deleting sub		 tensors " << std::endl;
		unsigned count = 1;
		for (auto it = SubTensorMap.begin(); it != SubTensorMap.end(); ++it) {
		//	std::cout << "deleted = " << count << std::endl;
			if (it->second) {
			delete it->second;
			it->second = nullptr;
			++count;
			}
		}
	}
	//std::cout << " success clear te" << std::endl;
}

//------------------------------------SUBTENSOR -------------------------------------//
//
//
template<typename number_type> Tensor<number_type>& Tensor<number_type>::operator()(Shape index, Shape shape) {
#ifndef BLACKCAT_DISABLE_RUNTIME_CHECKS

#endif

	std::pair<Shape, Shape> PairIndex(index, shape);


	if (SubTensorMap.find(PairIndex) == SubTensorMap.end()) {
		SubTensorMap[PairIndex] = generateSubTensor(index, shape);
		return *(SubTensorMap[PairIndex]);
	} else {
		return *(SubTensorMap[PairIndex]);
	}
}

template<typename number_type> const Tensor<number_type>& Tensor<number_type>::operator()(Shape index, Shape shape) const {
#ifndef BLACKCAT_DISABLE_RUNTIME_CHECKS

#endif
	std::pair<Shape, Shape> PairIndex(index, shape);


	if (SubTensorMap.find(PairIndex) == SubTensorMap.end()) {
		SubTensorMap[PairIndex] = generateSubTensor(index, shape);
		return *(SubTensorMap[PairIndex]);
	} else {
		return *(SubTensorMap[PairIndex]);
	}
}
template<typename number_type> Tensor<number_type>* Tensor<number_type>::generateSubTensor(Shape index, Shape shape) const {
	return new Tensor<number_type>(*this, index, shape);
}


//------------------------------Constructor for generating a sub tenso (NOT AN INDEX TENSOR)
namespace meh{
void print(Shape data, std::string message) {
	std::cout << message << ": ";
	for (int i = 0; i < data.size(); ++i)  {
		std::cout << data[i] << " " ;
	}
}
}

template<typename number_type>
Tensor<number_type>::Tensor(const Tensor<number_type>& super_tensor, Shape index, Shape shape)
:
tensor_ownership(false), rank_ownership(true), ld_ownership(false), subTensor(true){
	if (super_tensor.order != index.size() || shape.size() > index.size()) {
		throw std::invalid_argument("Non-definitive index given for subTensor");
	}
	for (unsigned i = 0; i < index.size(); ++i) {
		if (index[index.size() - 1 - i]  > super_tensor.ranks[i]) {
			super_tensor.printDimensions();
					meh::print(index, "index " );
					meh::print(shape, " shape");
					throw std::invalid_argument("sub index out of bounds");

			}
		}

	for (unsigned i = 0; i < shape.size(); ++i) {
		if (shape[i] + index[index.size() - 1 - i] > super_tensor.ranks[i]) {
			super_tensor.printDimensions();
			meh::print(index, " index" );
			meh::print(shape, " shape");
			throw std::invalid_argument("sub tensor size greater than primary matrix");
		}

	}

	parent = &super_tensor;
	sz = BC::calc_sz(&shape[0], shape.size());
	order = shape.size();
	ranks = new unsigned[order];
	ld = super_tensor.ld;
	BC::copy(ranks, &shape[0], order);

	unsigned t_id = 0;
	for (int i = 0; i < index.size(); ++i) {
		t_id += super_tensor.ld[i] * index[index.size() - i - 1];
	}
	tensor = &super_tensor.tensor[t_id];
}

























