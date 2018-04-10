#include "BLACKCAT_CPU_MATHEMATICS.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

//will switch to Blas eventually
//template<typename number_type>
//void Tensor_Operations<number_type>::dot(number_type * s, const number_type * m1, unsigned m1_rows, unsigned m1_cols, const number_type * m2, unsigned m2_rows, unsigned m2_cols) {
//	Tensor_Operations<number_type>::fill(s, number_type(), m1_rows * m2_cols);
//
//    for (unsigned r1 = 0; r1 < m1_rows; ++r1) {
//        for (unsigned shrd_side = 0; shrd_side < m1_cols; ++shrd_side) {
//            for (unsigned c2 = 0; c2 < m2_cols; ++c2) {
//                unsigned id_s = c2 + r1 * m2_cols;
//                unsigned id_1 = r1 * m1_cols + shrd_side;
//                unsigned id_2 = shrd_side * m1_rows + c2;
//
//                s[id_s] += m1[id_1] * m2[id_2];
//            }
//        }
//    }
//}
//???may not work
//template<typename number_type>
//void Tensor_Operations<number_type>::dot_outerproduct(number_type* s, const number_type* m1, unsigned m1_sz, const number_type* m2, unsigned m2_sz) {
//	for (int r = 0; r < m1_sz; ++r) {
//		unsigned row = r * m1_sz;
//		for (int c = 0; c < m2_sz; ++c) {
//			s[row + c] = m1[r] * m1[c];
//		}
//	}
//}
//
//template<>
//void Tensor_Operations<float>::dot(float * s, const float * m1, unsigned m1_rows, unsigned m1_cols, const float * m2, unsigned m2_rows, unsigned m2_cols) {
//	Tensor_Operations<float>::fill(s, 0, m1_rows * m2_cols);
//
//	//rowmajor
////	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
////            m1_rows, m2_cols, m1_cols,
////            1, m1, m1_cols,
////            m2, m2_cols, 1,
////            s, m2_cols);
////col major
//	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
//            m1_rows, m2_cols, m1_cols,
//            1, m1, m1_rows,
//            m2, m2_rows, 1,
//            s, m1_rows);
//}
//
//
//template<>
//void Tensor_Operations<double>::dot(double * s, const double * m1, unsigned m1_rows, unsigned m1_cols, const double * m2, unsigned m2_rows, unsigned m2_cols) {
//	Tensor_Operations<double>::fill(s, 0, m1_rows * m2_cols);
//
//
////	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
////            m1_rows, m2_cols, m1_cols,
////            1, m1, m1_cols,
////            m2, m2_cols, 1,
////            s, m2_cols);
//
//	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
//            m1_rows, m2_cols, m1_cols,
//            1, m1, m1_rows,
//            m2, m2_rows, 1,
//            s, m1_rows);
//}
//


//template<typename number_type>
//void Tensor_Operations<number_type>::dot(number_type * s, unsigned s_LD, const number_type * m1, unsigned m1_rows, unsigned m1_cols, unsigned m1_LD,
//											  const number_type * m2, unsigned m2_rows, unsigned m2_cols, unsigned m2_LD) {
//	throw std::invalid_argument("dot not supported on non -float/double types");
//}
template<>
void CPU_MATHEMATICS<float>::dot(float * s, unsigned s_LD, const float * m1, unsigned m1_rows, unsigned m1_cols, unsigned m1_LD,
											  const float * m2, unsigned m2_rows, unsigned m2_cols, unsigned m2_LD) {
	CPU_MATHEMATICS<float>::fill(s, 0, m1_rows * m2_cols);

	//rowmajor
//	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
//            m1_rows, m2_cols, m1_cols,
//            1, m1, m1_cols,
//            m2, m2_cols, 1,
//            s, m2_cols);
//col major
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            m1_rows, m2_cols, m1_cols,
            1, m1, m1_LD,
            m2, m2_LD, 1,
            s, s_LD);
}

template<>
void CPU_MATHEMATICS<double>::dot(double * s, unsigned s_LD, const double * m1, unsigned m1_rows, unsigned m1_cols, unsigned m1_LD,
											  const double * m2, unsigned m2_rows, unsigned m2_cols, unsigned m2_LD) {
	CPU_MATHEMATICS<double>::fill(s, 0, m1_rows * m2_cols);

	//rowmajor
//	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
//            m1_rows, m2_cols, m1_cols,
//            1, m1, m1_cols,
//            m2, m2_cols, 1,
//            s, m2_cols);
//col major
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            m1_rows, m2_cols, m1_cols,
            1, m1, m1_LD,
            m2, m2_LD, 1,
            s, s_LD);
}



template<typename number_type>
void CPU_MATHEMATICS<number_type>::dot(number_type* store, unsigned s_inc, const number_type* m1, unsigned m1_r, unsigned m1_c, unsigned m1_inc,
																			 const number_type* m2, unsigned m2_r, unsigned m2_c, unsigned m2_inc)
{
	throw std::invalid_argument("dot product not supported on non-double/float types");
}

