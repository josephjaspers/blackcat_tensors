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
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

namespace GPU_MATHEMATICS {

	//Memory Management [written for preparation for writing with CUDA]]
	 template<typename number_type> static void initialize	(number_type*& d, unsigned sz) {
		 cudaMalloc((void**)&d, sizeof(number_type)* sz);
	 }
	 template<typename number_type> static void destruction	(number_type* d) {
		 cudaFree(d);
	 }
	 template<typename number_type> static void initialize_unified(number_type*& d, unsigned sz) {
		 cudaMallocManaged((void**)&d, sizeof(number_type)* sz);
	 }

	//Pure pointwise methods (no LD)
      template<typename number_type> __global__ static void copy		(number_type* store, const number_type* v, unsigned sz);
      template<typename number_type> __global__ static void fill		(number_type* m, number_type f, unsigned sz);
      template<typename number_type> __global__ static void randomize	(number_type* m, number_type lower_bound, number_type upper_bound, unsigned sz);
      template<typename number_type> __global__ static void print 		(const number_type* m, unsigned sz);

    //Advanced -- no padding
      template<typename number_type> __global__ static void correlation(number_type* s, const number_type* filter, const number_type* signal, unsigned sz);

    //Pointwise
      template<typename number_type> __global__ static void power		(number_type* s, const number_type* m1, const number_type* m2, unsigned sz);
      template<typename number_type> __global__ static void divide		(number_type* s, const number_type* m1, const number_type* m2, unsigned sz);
      template<typename number_type> __global__ static void add			(number_type* s, const number_type* m1, const number_type* m2, unsigned sz);
      template<typename number_type> __global__ static void subtract	(number_type* s, const number_type* m1, const number_type* m2, unsigned sz);
      template<typename number_type> __global__ static void multiply	(number_type* s, const number_type* m1, const number_type* m2, unsigned sz);
    //Pointwise Scalar
      template<typename number_type> __global__ static void power		(number_type *s, const number_type *m, number_type v, unsigned sz);
      template<typename number_type> __global__ static void divide		(number_type *s, const number_type *m, number_type v, unsigned sz);
      template<typename number_type> __global__ static void add			(number_type *s, const number_type *m, number_type v, unsigned sz);
      template<typename number_type> __global__ static void subtract	(number_type *s, const number_type *m, number_type v, unsigned sz);
      template<typename number_type> __global__ static void multiply	(number_type *s, const number_type *m, number_type v, unsigned sz);

    //---Primary Indexers
      template<typename number_type> __global__ static void max(const number_type* m, unsigned* ld, unsigned* ranks, unsigned order, number_type* max_val);
      template<typename number_type> __global__ static void min(const number_type* m, unsigned* ld, unsigned* ranks, unsigned order, number_type* max_val);

      template<typename number_type> __global__ static void max_index(number_type* min_val, unsigned* min_indexes, const number_type* data, const unsigned* ranks, const unsigned* ld, unsigned order);
      template<typename number_type> __global__ static void min_index(number_type* min_val, unsigned* min_indexes, const number_type* data, const unsigned* ranks, const unsigned* ld, unsigned order);

   // Pointwise_ LD -scaleable variants - - - - - - - - -COLMAJOR- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      template<typename number_type> __global__ static void dot		(number_type* store, unsigned store_LD, const number_type* m1, unsigned m1_row, unsigned m1_col, unsigned m1_inc,
    															const number_type* m2, unsigned m2_row, unsigned m2_col, unsigned m2_inc);
    //Pointwise increments ------------- COL Major --- for degree > 2 Ten
      template<typename number_type> __global__ static void copy     (number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1, const unsigned* m1_LD);
      template<typename number_type> __global__ static void fill     (number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, number_type m1);
      template<typename number_type> __global__ static void transpose(number_type* s, unsigned s_ld, const number_type* m, unsigned r, unsigned c, unsigned m_ld);
      template<typename number_type> __global__ static void randomize(number_type* s, unsigned* s_ld, unsigned* ranks, unsigned order, number_type lb, number_type ub);
      template<typename number_type> __global__ static void print		(const number_type* ary, const unsigned* dims, const unsigned* lead_dims, unsigned index);

    //Advanced operations
      template<typename number_type> __global__ static void axpy(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,
    		 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	  const unsigned* m1_LD, number_type scalar);


      template<typename number_type> __global__ static void correlation(number_type* s, unsigned order, const unsigned* ranks,const number_type* filter, const unsigned* f_ld,
    																		      const number_type* signal, const unsigned* s_ld);

      template<typename number_type> __global__ static void cross_correlation(number_type* s, unsigned cor_mv, const  unsigned* store_ld, const number_type* filter,const  unsigned * f_ld,const  unsigned* f_ranks, unsigned f_order,
    																								const number_type* signal, const unsigned * s_ld, const unsigned* s_ranks, unsigned s_order);


      template<typename number_type> __global__ static void axpy(number_type* store, const unsigned* store_ld, const number_type* signal, const unsigned* signal_ld, const unsigned* signal_ranks, unsigned signal_order, number_type scalar);

      template<typename number_type> __global__ static void neg_axpy(number_type* store, const unsigned* store_ld, const number_type* signal, const unsigned* signal_ld, const unsigned* signal_ranks, unsigned signal_order, number_type scalar);

      template<typename number_type> __global__ static void cc_filter_error(unsigned move_dimensions, number_type* store, const unsigned* store_ld, const unsigned* store_ranks, unsigned store_order,
    		 const  number_type* error, const unsigned* error_ld, const unsigned* error_ranks, unsigned error_order,
    		 const  number_type* signal,const unsigned* signal_ld, const unsigned* signal_ranks, unsigned signal_order);

      template<typename number_type> __global__ static void cc_signal_error(unsigned move_dimensions, number_type* store, const unsigned* store_ld, const unsigned* store_ranks, unsigned store_order,
         		 const  number_type* error, const unsigned* error_ld, const unsigned* error_ranks, unsigned error_order,
         		 const  number_type* signal,const unsigned* signal_ld, const unsigned* signal_ranks, unsigned signal_order);

     //Pointwise
      template<typename number_type> __global__ static void power	(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type* m2,  const unsigned* m2_LD);
      template<typename number_type> __global__ static void multiply(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type* m2,  const unsigned* m2_LD);
      template<typename number_type> __global__ static void divide	(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type* m2,  const unsigned* m2_LD);
      template<typename number_type> __global__ static void add		(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type* m2,  const unsigned* m2_LD);
      template<typename number_type> __global__ static void subtract(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type* m2,  const unsigned* m2_LD);
    //By Scalar
      template<typename number_type> __global__ static void power	(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type scal);
      template<typename number_type> __global__ static void multiply(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type scal);
      template<typename number_type> __global__ static void divide	(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type scal);
      template<typename number_type> __global__ static void add		(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type scal);
      template<typename number_type> __global__ static void subtract(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type scal);

      template<typename number_type> __global__ static void max(number_type* max, const number_type* data, const unsigned *ranks, const unsigned* ld, unsigned order);
      template<typename number_type> __global__ static void min(number_type* min, const number_type* data, const unsigned *ranks, const unsigned* ld, unsigned order);
};
#endif /* LINEARALGEBRAROUTINES_H */

