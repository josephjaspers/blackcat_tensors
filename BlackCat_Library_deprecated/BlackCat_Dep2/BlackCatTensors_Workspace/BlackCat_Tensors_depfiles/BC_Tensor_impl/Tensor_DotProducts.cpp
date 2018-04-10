#include "Tensor.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename number_type>
Tensor<number_type>::Tensor(const Tensor<number_type>& m1, const Tensor<number_type>& m2) : ownership(true) {
	//DOTPRODUCT CONSTRUCTOR
	order = m2.order;
	ranks = new unsigned[order];

	BC::copy(ranks, m2.ranks, order);

	if (order > 1)
	ranks[1] = m1.rows();

	ranks[0] = m2.cols();

	sz = (m2.size() / m2.rows()) * m1.rows();
	Tensor_Operations<number_type>::initialize(tensor, sz);
}



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
        Tensor_Operations<number_type>::dot(&s.tensor[s_index], &tensor[m1_index], this->rows(), this->cols(), &t.tensor[m2_index], t.cols());
    }
    return s;
}

