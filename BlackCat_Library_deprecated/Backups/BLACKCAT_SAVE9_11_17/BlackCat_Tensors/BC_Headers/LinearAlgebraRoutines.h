/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   LinearAlgebraRoutines.h
 * Author: joseph
 *
 * Created on July 23, 2017, 9:11 PM
 */

#ifndef LINEARALGEBRAROUTINES_H
#define LINEARALGEBRAROUTINES_H
#include <iostream>
#include <math.h>
#include "cblas.h"

template <typename number_type>
class Tensor_Operations {
public:
    //Advanced -- no padding
    static void correlation(number_type* s, const number_type* filter, const number_type* signal, unsigned sz);
    static void correlation(number_type* s, unsigned order, const unsigned* ranks,const number_type* filter, const unsigned* f_ld,
    																		      const number_type* signal, const unsigned* s_ld);
    //dimensional --movement correlation
    static void cross_correlation(number_type* s, unsigned cor_mv, unsigned order, const  unsigned* store_ld, const number_type* filter,const  unsigned * f_ld,const  unsigned* f_ranks,
    																										  const number_type* signal, const unsigned * s_ld, const unsigned* s_ranks);
    //Memory Management [written for preparation for writing with CUDA]]
    static void initialize	(number_type*& d, unsigned sz);
    static void destruction	(number_type* d);

    //Pure pointwise methods (no LD)
    static void copy		(number_type* store, const number_type* v, unsigned sz);
    static void fill		(number_type* m, number_type f, unsigned sz);
    static void randomize	(number_type* m, int lower_bound, int upper_bound, unsigned sz);
    //Pointwise
    static void power		(number_type* s, const number_type* m1, const number_type* m2, unsigned sz);
    static void divide		(number_type* s, const number_type* m1, const number_type* m2, unsigned sz);
    static void add			(number_type* s, const number_type* m1, const number_type* m2, unsigned sz);
    static void subtract	(number_type* s, const number_type* m1, const number_type* m2, unsigned sz);
    static void multiply	(number_type* s, const number_type* m1, const number_type* m2, unsigned sz);
    //Pointwise Scalar
    static void power		(number_type *s, const number_type *m, number_type v, unsigned sz);
    static void divide		(number_type *s, const number_type *m, number_type v, unsigned sz);
    static void add			(number_type *s, const number_type *m, number_type v, unsigned sz);
    static void subtract	(number_type *s, const number_type *m, number_type v, unsigned sz);
    static void multiply	(number_type *s, const number_type *m, number_type v, unsigned sz);
    //Indexing
    static void max_val		(const number_type* m, number_type* max_val, unsigned sz);
    static void min_val		(const number_type* m, number_type* min_val, unsigned sz);
    static void max_index	(const number_type * m, number_type * max_val, unsigned * index, unsigned sz);
    static void min_index	(const number_type * m, number_type * min_val, unsigned * index, unsigned sz);

   // Pointwise_ LD -scaleable variants - - - - - - - - -COLMAJOR- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    static void dot		(number_type* store, unsigned store_LD, const number_type* m1, unsigned m1_row, unsigned m1_col, unsigned m1_inc,
    															const number_type* m2, unsigned m2_row, unsigned m2_col, unsigned m2_inc);

    //Pointwise increments ------------- COL Major --- for degree > 2 Ten
    static void copy     (number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1, const unsigned* m1_LD);
    static void fill     (number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, number_type m1);
    static void transpose(number_type* s, unsigned s_ld, const number_type* m, unsigned r, unsigned c, unsigned m_ld);

    static void power	(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type* m2,  const unsigned* m2_LD);
    static void multiply(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type* m2,  const unsigned* m2_LD);
    static void divide	(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type* m2,  const unsigned* m2_LD);
    static void add		(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type* m2,  const unsigned* m2_LD);
    static void subtract(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type* m2,  const unsigned* m2_LD);
    //By Scalar
    static void power	(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type scal);
    static void multiply(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type scal);
    static void divide	(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type scal);
    static void add		(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type scal);
    static void subtract(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type scal);

};
#endif /* LINEARALGEBRAROUTINES_H */

