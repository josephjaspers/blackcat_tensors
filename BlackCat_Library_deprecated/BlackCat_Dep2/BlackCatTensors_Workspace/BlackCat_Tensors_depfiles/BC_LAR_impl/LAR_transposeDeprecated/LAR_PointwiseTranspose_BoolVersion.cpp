

#include "LinearAlgebraRoutines.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename number_type>
void Tensor_Operations<number_type>::multiply(number_type* s, const number_type* m1, bool transposed1, const number_type* m2, bool transposed2, unsigned rows, unsigned cols) {

    unsigned sz = rows * cols;

    //Both are transposed
    if (transposed1 && transposed2) {
        for (unsigned i = 0; i < sz; ++i) {
            unsigned index_mat1 = i % rows * cols + (int) floor(i / rows) % cols;
            unsigned index_mat2 = i % rows * cols + (int) floor(i / rows) % cols;
            s[i] = m1[index_mat1] * m2[index_mat2];
        }
    } else if (transposed1) {
        for (unsigned i = 0; i < sz; ++i) {
            unsigned index_mat1 = i % rows * cols + (int) floor(i / rows) % cols;
            s[i] = m1[index_mat1] * m2[i];
        }
    } else if (transposed2) {
        for (unsigned i = 0; i < sz; ++i) {

            unsigned index_mat2 = i % rows * cols + (int) floor(i / rows) % cols;
            s[i] = m1[i] * m2[index_mat2];
        }
    } else
        for (unsigned i = 0; i < sz; ++i) {
            s[i] = m1[i] * m2[i];
        }
}

template<typename number_type>
void Tensor_Operations<number_type>::divide(number_type* s, const number_type* m1, bool transposed1, const number_type* m2, bool transposed2, unsigned rows, unsigned cols) {

    unsigned sz = rows * cols;

    //Both are transposed
    if (transposed1 && transposed2) {
        for (unsigned i = 0; i < sz; ++i) {
            unsigned index_mat1 = i % rows * cols + (int) floor(i / rows) % cols;
            unsigned index_mat2 = i % rows * cols + (int) floor(i / rows) % cols;
            s[i] = m1[index_mat1] / m2[index_mat2];
        }
    } else if (transposed1) {
        for (unsigned i = 0; i < sz; ++i) {
            unsigned index_mat1 = i % rows * cols + (int) floor(i / rows) % cols;
            s[i] = m1[index_mat1] / m2[i];
        }
    } else if (transposed2) {
        for (unsigned i = 0; i < sz; ++i) {

            unsigned index_mat2 = i % rows * cols + (int) floor(i / rows) % cols;
            s[i] = m1[i] / m2[index_mat2];
        }
    } else
        for (unsigned i = 0; i < sz; ++i) {
            s[i] = m1[i] * m2[i];
        }
}

template<typename number_type>
void Tensor_Operations<number_type>::add(number_type* s, const number_type* m1, bool transposed1, const number_type* m2, bool transposed2, unsigned rows, unsigned cols) {

    unsigned sz = rows * cols;

    //Both are transposed
    if (transposed1 && transposed2) {
        for (unsigned i = 0; i < sz; ++i) {
            unsigned index_mat1 = i % rows * cols + (int) floor(i / rows) % cols;
            unsigned index_mat2 = i % rows * cols + (int) floor(i / rows) % cols;
            s[i] = m1[index_mat1] + m2[index_mat2];
        }
    } else if (transposed1) {
        for (unsigned i = 0; i < sz; ++i) {
            unsigned index_mat1 = i % rows * cols + (int) floor(i / rows) % cols;
            s[i] = m1[index_mat1] + m2[i];
        }
    } else if (transposed2) {
        for (unsigned i = 0; i < sz; ++i) {

            unsigned index_mat2 = i % rows * cols + (int) floor(i / rows) % cols;
            s[i] = m1[i] * m2[index_mat2];
        }
    } else
        for (unsigned i = 0; i < sz; ++i) {
            s[i] = m1[i] + m2[i];
        }
}

template<typename number_type>
void Tensor_Operations<number_type>::subtract(number_type* s, const number_type* m1, bool transposed1, const number_type* m2, bool transposed2, unsigned rows, unsigned cols) {

    unsigned sz = rows * cols;

    //Both are transposed
    if (transposed1 && transposed2) {
        for (unsigned i = 0; i < sz; ++i) {
            unsigned index_mat1 = i % rows * cols + (int) floor(i / rows) % cols;
            unsigned index_mat2 = i % rows * cols + (int) floor(i / rows) % cols;
            s[i] = m1[index_mat1] - m2[index_mat2];
        }
    } else if (transposed1) {
        for (unsigned i = 0; i < sz; ++i) {
            unsigned index_mat1 = i % rows * cols + (int) floor(i / rows) % cols;
            s[i] = m1[index_mat1] - m2[i];
        }
    } else if (transposed2) {
        for (unsigned i = 0; i < sz; ++i) {

            unsigned index_mat2 = i % rows * cols + (int) floor(i / rows) % cols;
            s[i] = m1[i] - m2[index_mat2];
        }
    } else
        for (unsigned i = 0; i < sz; ++i) {
            s[i] = m1[i] - m2[i];
        }
} 
