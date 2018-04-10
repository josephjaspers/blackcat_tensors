#include "Tensor.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename number_type, class TensorOperations>
Tensor<number_type, TensorOperations> Tensor<number_type, TensorOperations>::operator*(const Tensor<number_type, TensorOperations>& t) const {
    this->assert_dotProduct_dimensions(t);
    Tensor<number_type, TensorOperations> s(*this, t);

    unsigned m1_matsz = matrix_size();
    unsigned m2_matsz = t.matrix_size();
    unsigned s_matsz = s.matrix_size();

    unsigned mat_total = totalMatrices();

    for (int i = 0; i < mat_total; ++i) {
        unsigned m1_index = m1_matsz * i;
        unsigned m2_index = m2_matsz * i;
        unsigned s_index = s_matsz * i;

        CPU::dot(&s.tensor[s_index], s.leading_dim(1), &tensor[m1_index], this->rows(), this->cols(), leading_dim(1),
        																		&t.tensor[m2_index],     t.rows(),     t.cols(), t.leading_dim(1));
    }
    return s;
}

template<typename number_type, class TensorOperations>
Scalar<number_type, TensorOperations> Tensor<number_type, TensorOperations>::corr(const Tensor<number_type, TensorOperations>& t) const  {
	assert_same_dimensions(t);
	Scalar<number_type, TensorOperations> scal(0);

	CPU::correlation(scal.data(), order, ranks, tensor, ld, t.tensor, ld);

	return scal;

}

template<typename number_type, class TensorOperations>
Tensor<number_type, TensorOperations> Tensor<number_type, TensorOperations>::x_corr_full_stack(unsigned move_dimensions, const Tensor<number_type, TensorOperations>& t) const {
	Shape buffer = t.getShape();
	for (unsigned i = 0 ; i < move_dimensions; ++i) {
		if (i < order - 1) {
			buffer[i] += (rank(i) - 1) * 2;
		}

	}
	Tensor<number_type, TensorOperations> buffer_tensor(buffer);

	Shape index_sub(t.order);
	for (unsigned i = 0; i < t.order; ++i) {
		if (i < move_dimensions){
			if (i < order - 1) {
				index_sub[t.order - i - 1] = rank(i) - 1;
			} else {
				index_sub[t.order - i - 1] = 0;
			}
		} else {
			index_sub[t.order - i - 1] = 0;
		}
	}
	buffer_tensor(index_sub, t.getShape()) = t;

	return x_corr_stack(move_dimensions, buffer_tensor);
}

template<typename number_type, class TensorOperations>
Tensor<number_type, TensorOperations> Tensor<number_type, TensorOperations>::x_corr_stack(unsigned dims, const Tensor<number_type, TensorOperations>& t) const {

	Shape x_shape(dims + 1);
	for (unsigned i = 0; i < dims; ++i) {
		if (i < order - 1)
		x_shape[i] = t.rank(i) - rank(i) + 1;
		else {
			x_shape[i] = t.rank(i);
		}
	}

	x_shape.back() = outerRank();
	Tensor<number_type, TensorOperations> output(x_shape); output.fill(0);

	for (unsigned i = 0; i < output.outerRank(); ++i) {
		CPU::cross_correlation(&output.tensor[i * output.outerLD()], dims, output.ld,
				&tensor[i * ld[order - 1]], ld, ranks, order - 1, t.tensor, t.ld, t.ranks, t.order);
		}
	return output;
}

//----------------

template<typename number_type, class TensorOperations>
Tensor<number_type, TensorOperations> Tensor<number_type, TensorOperations>::x_corr_full(unsigned move_dimensions, const Tensor<number_type, TensorOperations>& t) const {
	unsigned max_order = t.order > order ? t.order : order;
	for (unsigned i = move_dimensions; i < max_order; ++i)
	{
		if (rank(i) != t.rank(i))
		{
			throw std::invalid_argument("rank mismatch on non motion _ xcorr_adj");
		}
	}

	Shape buffer = t.getShape();
	for (unsigned i = 0; i < move_dimensions; ++i)
	{
		buffer[i] += (rank(i) - 1) * 2;
	}
	Tensor<number_type, TensorOperations> buffer_tensor(buffer);

	Shape index_sub(t.order);
	for (unsigned i = 0; i < t.order; ++i)
	{
		if (i < move_dimensions)
		{
			index_sub[t.order - i - 1] = rank(i) - 1;
		} else
		{
			index_sub[t.order - i - 1] = 0;
		}
	}
	buffer_tensor.fill(0);
	buffer_tensor(index_sub, t.getShape()) = t;

	return x_corr(move_dimensions, buffer_tensor);
}

template<typename number_type, class TensorOperations> //valid x_corr
Tensor<number_type, TensorOperations> Tensor<number_type, TensorOperations>::x_corr(unsigned dims, const Tensor<number_type, TensorOperations>& t) const {

	unsigned max_order = t.order > order ? t.order : order;
	for (unsigned i = dims; i < max_order; ++i)
	{
		if (rank(i) != t.rank(i))
		{
			throw std::invalid_argument("rank mismatch on non motion _ xcorr_adj");
		}
	}
	for (unsigned i = 0; i < dims ; ++i) {
		if (rank(i) > t.rank(i)) {
			throw std::invalid_argument("valid correlation filter must have a smaller dimenisonality than the signal within the movement dimensions");
		}
	}

	//create the dimensionality of the output tensor
	Shape x_shape(dims);
	for (unsigned i = 0; i < dims; ++i)
	{
		x_shape[i] = t.rank(i) - rank(i) + 1;
	}

	Tensor<number_type, TensorOperations> output(x_shape); output.fill(0);
	CPU::cross_correlation(output.tensor, dims, output.ld, tensor, ld, ranks, order,
	t.tensor, t.ld, t.ranks, t.order);
	return output;
}

template<typename number_type, class TensorOperations> //valid x_corr
Tensor<number_type, TensorOperations> Tensor<number_type, TensorOperations>::x_corr_FilterError(unsigned move_dimensions, const Tensor<number_type, TensorOperations>& error) const {

	//this = the signal  t is the error
	Shape original_filter_shape(this->degree()); //same dimensions
	for (unsigned i = 0; i < degree(); ++i) {
		if (i < move_dimensions)
			original_filter_shape[i] = this->ranks[i] - error.ranks[i] + 1;
		else
			original_filter_shape[i] = this->ranks[i];
	}
	Tensor<number_type, TensorOperations> output(original_filter_shape);
	output = 0;
	CPU::cross_correlation_filter_error(move_dimensions, output.tensor, output.ld, output.ranks, output.order,
													error.tensor, error.ld, error.ranks, error.order,
														  tensor, ld, ranks, order);
	return output;
}

template<typename number_type, class TensorOperations> //valid x_corr
Tensor<number_type, TensorOperations> Tensor<number_type, TensorOperations>::x_corr_SignalError(unsigned move_dimensions, const Tensor<number_type, TensorOperations>& error) const {

	//this = the weights  t is the error
	Shape original_filter_shape(this->degree()); //same dimensions
	for (unsigned i = 0; i < degree(); ++i) {
		if (i < move_dimensions)
			original_filter_shape[i] = this->rank(i) + error.rank(i) - 1;
		else
			original_filter_shape[i] = this->rank(i);
	}
	Tensor<number_type, TensorOperations> output(original_filter_shape);
	output = 0;

	CPU::cross_correlation_signal_error(move_dimensions, output.tensor, output.ld, output.ranks, output.order,
													error.tensor, error.ld, error.ranks, error.order,
														  tensor, ld, ranks, order);
	return output;
}
