#include "Tensor.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename number_type>
Tensor<number_type> Tensor<number_type>::operator*(const Tensor<number_type>& t) const {
    this->assert_dotProduct_dimensions(t);
    Tensor<number_type> s(*this, t);

    unsigned m1_matsz = matrix_size();
    unsigned m2_matsz = t.matrix_size();
    unsigned s_matsz = s.matrix_size();

    unsigned mat_total = totalMatrices();

    for (int i = 0; i < mat_total; ++i) {
        unsigned m1_index = m1_matsz * i;
        unsigned m2_index = m2_matsz * i;
        unsigned s_index = s_matsz * i;
        Tensor_Operations<number_type>::dot(&s.tensor[s_index], &tensor[m1_index], this->rows(), this->cols(), &t.tensor[m2_index], t.rows(), t.cols());
    }
    return s;
}

template<typename number_type>
Tensor<number_type> Tensor<number_type>::operator->*(const Tensor<number_type>& t) const {
	Tensor<number_type> r(this->size(), t.size());
	Tensor_Operations<number_type>::dot(r.data(), this->data(), this->size(), 1,  t.data(), 1, t.size());
	return r;
}
template<typename number_type>
Tensor<number_type> Tensor<number_type>::convolution(const Tensor<number_type>& t) const {
	assert_valid_convolution_target(t);
	Tensor<number_type> conv(t.rows() - rows(), t.cols() - cols());
	Tensor_Operations<number_type>::convolution(conv.tensor, tensor, rows(), t.tensor,t.rows());
	return conv;
}

template<typename number_type>
Tensor<number_type> Tensor<number_type>::correlation(const Tensor<number_type>& t) const {
	assert_valid_correlation_target(t);
	Tensor<number_type> conv(t.rows() - rows(), t.cols() - cols());
	Tensor_Operations<number_type>::correlation(conv.tensor, tensor, rows(), t.tensor,t.rows());
	return conv;
}
