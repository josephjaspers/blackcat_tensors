#include "LinearAlgebraRoutines.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

//will switch to Blas eventually 

template<typename number_type>
void Tensor_Operations<number_type>::dot(number_type * s, const number_type * m1, unsigned m1_rows, unsigned m1_cols, const number_type * m2, unsigned m2_rows, unsigned m2_cols) {
	Tensor_Operations<number_type>::fill(s, number_type(), m1_rows * m2_cols);

    for (unsigned r1 = 0; r1 < m1_rows; ++r1) {
        for (unsigned shrd_side = 0; shrd_side < m1_cols; ++shrd_side) {
            for (unsigned c2 = 0; c2 < m2_cols; ++c2) {
                unsigned id_s = c2 + r1 * m2_cols;
                unsigned id_1 = r1 * m1_cols + shrd_side;
                unsigned id_2 = shrd_side * m1_rows + c2;

                s[id_s] += m1[id_1] * m2[id_2];
            }
        }
    }
}
//???may not work
template<typename number_type>
void Tensor_Operations<number_type>::dot_outerproduct(number_type* s, const number_type* m1, unsigned m1_sz, const number_type* m2, unsigned m2_sz) {
	for (int r = 0; r < m1_sz; ++r) {
		unsigned row = r * m1_sz;
		for (int c = 0; c < m2_sz; ++c) {
			s[row + c] = m1[r] * m1[c];
		}
	}
}

template<>
void Tensor_Operations<float>::dot(float * s, const float * m1, unsigned m1_rows, unsigned m1_cols, const float * m2, unsigned m2_rows, unsigned m2_cols) {
	Tensor_Operations<float>::fill(s, 0, m1_rows * m2_cols);

	//rowmajor
//	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
//            m1_rows, m2_cols, m1_cols,
//            1, m1, m1_cols,
//            m2, m2_cols, 1,
//            s, m2_cols);
//col major
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            m1_rows, m2_cols, m1_cols,
            1, m1, m1_rows,
            m2, m2_rows, 1,
            s, m1_rows);
}


template<>
void Tensor_Operations<double>::dot(double * s, const double * m1, unsigned m1_rows, unsigned m1_cols, const double * m2, unsigned m2_rows, unsigned m2_cols) {
	Tensor_Operations<double>::fill(s, 0, m1_rows * m2_cols);


//	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
//            m1_rows, m2_cols, m1_cols,
//            1, m1, m1_cols,
//            m2, m2_cols, 1,
//            s, m2_cols);

	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            m1_rows, m2_cols, m1_cols,
            1, m1, m1_rows,
            m2, m2_rows, 1,
            s, m1_rows);
}
//
//template<typename number_type>
//void Tensor_Operations<number_type>::dot_transposeA(number_type * s, const number_type * m1, unsigned m1_rows, unsigned m1_cols, const number_type * m2, unsigned m2_rows, unsigned m2_cols) {
//	throw std::invalid_argument("tranposeA not supported");
//}
//
//template<>
//void Tensor_Operations<float>::dot_transposeA(float * s, const float * m1, unsigned m1_rows, unsigned m1_cols, const float * m2, unsigned m2_rows, unsigned m2_cols) {
//	Tensor_Operations<float>::fill(s, 0, m1_rows * m2_cols);
//
//
//	float* m1_t = new float[m1_rows * m1_cols];
//	Tensor_Operations<float>::transpose(m1_t, m1, m1_rows, m1_cols); 	//flip rows and cols (as parameters are the "new" dimensions);
//
//    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
//            m1_rows, m2_cols, m1_cols,
//            1, m1_t, m1_cols,
//            m2, m2_cols, 1,
//            s, m2_cols);
//
//    delete[] m1_t;
//}
//
//template<>
//void Tensor_Operations<double>::dot_transposeA(double * s, const double * m1, unsigned m1_rows, unsigned m1_cols, const double * m2, unsigned m2_rows, unsigned m2_cols) {
//	Tensor_Operations<double>::fill(s, 0, m1_rows * m2_cols);
//	double* m1_t = new double[m1_rows * m1_cols];
//	Tensor_Operations<double>::transpose(m1_t, m1, m1_cols, m1_rows);	//flip rows and cols (as parameters are the "new" dimensions);
//
//    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
//            m1_rows, m2_cols, m1_cols,
//            1, m1_t, m1_cols,
//            m2, m2_cols, 1,
//            s, m2_cols);
//
//    delete[] m1_t;
//}
//
//template<typename number_type>
//void Tensor_Operations<number_type>::dot_transposeB(number_type * s, const number_type * m1, unsigned m1_rows, unsigned m1_cols, const number_type * m2, unsigned m2_rows, unsigned m2_cols) {
//	throw std::invalid_argument("tranposeB not supported");
//}
//
//template<>
//void Tensor_Operations<float>::dot_transposeB(float * s, const float * m1, unsigned m1_rows, unsigned m1_cols, const float * m2, unsigned m2_rows, unsigned m2_cols) {
//	Tensor_Operations<float>::fill(s, 0, m1_rows * m2_cols);
//
//	float* m2_t = new float[m2_rows * m2_cols];
//	Tensor_Operations<float>::transpose(m2_t, m2, m2_cols, m2_rows); 	//flip rows and cols (as parameters are the "new" dimensions);
//    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
//            m1_rows, m2_cols, m1_cols,
//            1, m1, m1_cols,
//            m2_t, m2_cols, 1,
//            s, m2_cols);
//
//    delete[] m2_t;
//}
//
//template<>
//void Tensor_Operations<double>::dot_transposeB(double * s, const double * m1, unsigned m1_rows, unsigned m1_cols, const double * m2, unsigned m2_rows, unsigned m2_cols) {
//	Tensor_Operations<double>::fill(s, 0, m1_rows * m2_cols);
//	double* m2_t = new double[m2_rows * m2_cols];
//	Tensor_Operations<double>::transpose(m2_t, m2, m2_cols, m2_rows); 	//flip rows and cols (as parameters are the "new" dimensions);
//
//	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
//            m1_rows, m2_cols, m1_cols,
//            1, m1, m1_cols,
//            m2_t, m2_cols, 1,
//            s, m2_cols);
//
//    delete[] m2_t;
//}
//
//template<typename number_type>
//void Tensor_Operations<number_type>::dot_transposeAB(number_type * s, const number_type * m1, unsigned m1_rows, unsigned m1_cols, const number_type * m2, unsigned m2_rows, unsigned m2_cols) {
//	throw std::invalid_argument("tranposeAB not supported");
//}
