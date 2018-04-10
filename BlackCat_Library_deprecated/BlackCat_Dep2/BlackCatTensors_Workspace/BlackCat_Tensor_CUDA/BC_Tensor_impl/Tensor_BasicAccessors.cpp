#include "Tensor.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename n, typename TensorOperations> bool Tensor<n, TensorOperations>::isInitialized() const
{
	return tensor;
}

template<typename n, typename TensorOperations> unsigned Tensor<n,
		TensorOperations>::totalMatrices() const
{
	unsigned total = 1;

	for (unsigned i = 2; i < order; ++i)
	{
		total *= ranks[i];
	}
	return total;
}

template<typename n, typename TensorOperations> unsigned Tensor<n,
		TensorOperations>::matrix_size() const
{
	return rows() * cols();
}
template<typename n, typename TensorOperations> unsigned Tensor<n,
		TensorOperations>::size() const
{
	return sz;
}
template<typename n, typename TensorOperations> unsigned Tensor<n,
		TensorOperations>::rows() const
{
	return order > 0 ? ranks[0] : 1;
}
template<typename n, typename TensorOperations> unsigned Tensor<n,
		TensorOperations>::cols() const
{
	return order > 1 ? ranks[1] : 1;
}
template<typename n, typename TensorOperations> unsigned Tensor<n,
		TensorOperations>::pages() const
{
	return order > 2 ? ranks[2] : 1;
}
template<typename n, typename TensorOperations> bool Tensor<n, TensorOperations>::isMatrix() const
{
	return order <= 2;
}
template<typename n, typename TensorOperations> bool Tensor<n, TensorOperations>::isVector() const
{
	return order <= 2 && (rows() == 1 || cols() == 1);
}
template<typename n, typename TensorOperations> bool Tensor<n, TensorOperations>::isSquare() const
{
	return order < 3 ? rows() == cols() : false;
}
template<typename n, typename TensorOperations> bool Tensor<n, TensorOperations>::isScalar() const
{
	return order == 0;
}
template<typename n, typename TensorOperations> unsigned Tensor<n,
		TensorOperations>::rank(unsigned rank_index) const
{
	return rank_index < degree() ? ranks[rank_index] : 1;
}

template<typename n, typename TensorOperations> unsigned Tensor<n,
		TensorOperations>::leading_dim(unsigned rank_index) const
{
	return rank_index < order ? ld[rank_index] : sz;
}
template<typename n, typename TensorOperations> unsigned Tensor<n,
		TensorOperations>::degree() const
{
	return order;
}

template<typename n, typename TensorOperations> unsigned Tensor<n,
		TensorOperations>::outerLD() const
{
	return ld[order - 1];
}
template<typename n, typename TensorOperations> unsigned Tensor<n,
		TensorOperations>::outerRank() const
{
	return ranks[order - 1];
}
template<typename n, typename TensorOperations> bool Tensor<n, TensorOperations>::needUpdate() const
{
	return needsUpdate;
}
template<typename n, typename TensorOperations> bool Tensor<n, TensorOperations>::owns_ranks() const
{
	return rank_ownership;
}
template<typename n, typename TensorOperations> bool Tensor<n, TensorOperations>::owns_tensor() const
{
	return tensor_ownership;
}
template<typename n, typename TensorOperations> bool Tensor<n, TensorOperations>::owns_LD() const
{
	return ld;
}
template<typename n, typename TensorOperations> bool Tensor<n, TensorOperations>::isSubTensor() const
{
	return subTensor;
}

