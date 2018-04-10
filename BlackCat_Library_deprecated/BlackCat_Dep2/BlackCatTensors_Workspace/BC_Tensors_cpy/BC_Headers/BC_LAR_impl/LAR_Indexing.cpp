/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   LAR_Indexing.h
 * Author: joseph
 *
 * Created on July 23, 2017, 3:24 PM
 */
#include "LinearAlgebraRoutines.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename number_type>
void Tensor_Operations<number_type>::max_val(const number_type * m, number_type * max_val, unsigned sz) {
    *max_val = m[0];


    for (int i = 1; i < sz; ++i) {
        if (*max_val < m[i]) {
            *max_val = m[i];
        }
    }
}

template<typename number_type>
void Tensor_Operations<number_type>::min_val(const number_type * m, number_type * min_val, unsigned sz) {
    *min_val = m[0];


    for (int i = 1; i < sz; ++i) {
        if (*min_val < m[i]) {
            *min_val = m[i];
        }
    }
}

template<typename number_type>
void Tensor_Operations<number_type>::max_index(const number_type * m, number_type * max_val, unsigned * index, unsigned sz) {
    *max_val = m[0];
    *index = 0;

    for (int i = 1; i < sz; ++i) {
        if (*max_val < m[i]) {
            *max_val = m[i];
            *index = i;
        }
    }
}

template<typename number_type>
void Tensor_Operations<number_type>::min_index(const number_type * m, number_type * min_val, unsigned * index, unsigned sz) {
    *min_val = m[0];
    *index = 0;

    for (int i = 1; i < sz; ++i) {
        if (*min_val < m[i]) {
            *min_val = m[i];
            *index = i;
        }
    }
}
