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

        Tensor_Operations<number_type>::dot(&s.tensor[s_index], s.leading_dim(1), &tensor[m1_index], this->rows(), this->cols(), leading_dim(1),
        																		&t.tensor[m2_index],     t.rows(),     t.cols(), t.leading_dim(1));
    }
    return s;
}
//
//template<typename number_type>
//Tensor<number_type> Tensor<number_type>::operator->*(const Tensor<number_type>& t) const {
//	Tensor<number_type> r(this->size(), t.size());
//  //  Tensor_Operations<number_type>::dot(r.tensor, r.leading_dim(1), tensor, this->size(), 1, this->size(), t.tensor, 1, t.size(), 1);
//
//		Tensor_Operations<number_type>::dot_outerproduct(r.data(), data(), t.data(), size(), t.size());
//	return r;
//}
template<typename number_type>
Scalar<number_type> Tensor<number_type>::corr(const Tensor<number_type>& t) const  {
	assert_same_dimensions(t);
	Scalar<number_type> scal(0);

	Tensor_Operations<number_type>::correlation(scal.data(), order, ranks, tensor, ld, t.tensor, ld);

	return scal;

}

template<typename number_type>
Tensor<number_type> Tensor<number_type>::x_corr(unsigned dims, const Tensor<number_type>& t) const {

	if (t.order != this->order && t.order != this->order - 1) {
		throw std::invalid_argument("x_cor dimenionsality mismatch");
	}
	for (unsigned i = dims; i < t.order; ++i) {
		if (ranks[i] != t.ranks[i]) {
			throw std::invalid_argument("rank mismatch on non motion");
		}
	}
	Shape x_shape(dims);
	for (unsigned i = 0; i < dims; ++i) {
		x_shape[i] = t.ranks[i] - ranks[i] + 1;
	}

	if (t.order == this->order) {
	Tensor<number_type> output(x_shape);
	Tensor_Operations<number_type>::cross_correlation(output.tensor, dims, order, output.ld, tensor, ld, ranks, t.tensor, t.ld, t.ranks);
	return output;
	} else {
		x_shape.push_back(ranks[order - 1]);
		Tensor<number_type> output(x_shape);

		for (unsigned i = 0; i < output.ranks[output.order - 1]; ++i) {
			Tensor_Operations<number_type>::cross_correlation(&output.tensor[i * output.ld[output.order - 1]], dims, order - 1, output.ld,
					&tensor[i * ld[order - 1]], ld, ranks, t.tensor, t.ld, t.ranks);

			}
		return output;
	}
}
