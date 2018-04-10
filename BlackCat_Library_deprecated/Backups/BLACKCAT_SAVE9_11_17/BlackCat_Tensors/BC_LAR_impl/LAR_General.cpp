/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   LAR_General.h
 * Author: joseph
 *
 * Created on July 23, 2017, 3:24 PM
 */

#include "LinearAlgebraRoutines.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename number_type>
void Tensor_Operations<number_type>::initialize(number_type*& d, unsigned sz) {
    d = new number_type[sz];
}

template<typename number_type>
void Tensor_Operations<number_type>::destruction(number_type * d) {
    delete [] d;
}

template<typename number_type>
void Tensor_Operations<number_type>::copy(number_type * store, const number_type * v, unsigned sz) {
    for (int i = 0; i < sz; ++i) {
        store[i] = v[i];
    }
}

template<typename number_type>
void Tensor_Operations<number_type>::fill(number_type * m, number_type f, unsigned sz) {
    for (int i = 0; i < sz; ++i) {
        m[i] = f;
    }
}

template<typename number_type>
void Tensor_Operations<number_type>::transpose(number_type * s, unsigned s_ld, const number_type * m, unsigned rows, unsigned cols, unsigned m_ld) {

	for (unsigned r = 0; r < rows; ++r) {
		for (unsigned c= 0; c< cols; ++c) {
			s[r * s_ld + c] = m[c * m_ld + r];
		}
	}
}

template<typename number_type>
void Tensor_Operations<number_type>::randomize(number_type * m, int lower_bound, int upper_bound, unsigned sz) {
    for (int i = 0; i < sz; ++i) {
        m[i] = rand() % (upper_bound - lower_bound) + lower_bound;
    }
}
