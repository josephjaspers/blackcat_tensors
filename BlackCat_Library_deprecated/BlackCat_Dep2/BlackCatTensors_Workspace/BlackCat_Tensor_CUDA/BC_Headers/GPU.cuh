/*
 * GPU.cuh
 *
 *  Created on: Sep 28, 2017
 *      Author: joseph
 */

#ifndef GPU_CUH_
#define GPU_CUHi_
#include "BLACKCAT_GPU_MATHEMATICS.cuh"

class GPU {
	   // Pointwise_ LD -scaleable variants - - - - - - - - -COLMAJOR- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

	template<typename number_type> static void dot(number_type* store, unsigned store_LD, const number_type* m1, unsigned m1_row, unsigned m1_col, unsigned m1_inc,
	    											const number_type* m2, unsigned m2_row, unsigned m2_col, unsigned m2_inc) {
		GPU_MATHEMATICS::dot<<<256,256>>>(store, store_LD, m1, m1_row, m1_col, m1_inc, m2, m2_row, m2_col, m2_inc);
	}

	template<typename number_type> static void max(number_type* max, const number_type* m1, const unsigned* ranks, const unsigned* ld, unsigned order) {

			GPU_MATHEMATICS::max<<<256,256>>>(max, m1, ranks, ld, order);
		}
	template<typename number_type> static void min(number_type* min, const number_type* m1, const unsigned* ranks, const unsigned* ld, unsigned order) {

			GPU_MATHEMATICS::min<<<256,256>>>(min, m1, ranks, ld, order);
		}

	template<typename number_type> static void min_index(number_type* min_val, unsigned* min_indexes, const number_type* data, const unsigned* ranks, const unsigned* ld, unsigned order) {
		GPU_MATHEMATICS::min_index<<<256,256>>>(min_val, min_indexes, data, ranks, ld, order);
	}
	template<typename number_type> static void max_index(number_type* min_val, unsigned* min_indexes, const number_type* data, const unsigned* ranks, const unsigned* ld, unsigned order) {
		GPU_MATHEMATICS::max_index<<<256,256>>>(min_val, min_indexes, data, ranks, ld, order);
	}

	 template<typename number_type> static void correlation	(number_type* s, const number_type* filter, const number_type* signal, unsigned sz) {
		GPU_MATHEMATICS::correlation<<<256,256>>>(s, filter, signal ,sz);
	}
	 template<typename number_type> static void correlation	(number_type* s, unsigned order, const unsigned* ranks,const number_type* filter, const unsigned* f_ld, const number_type* signal, const unsigned* s_ld) {
		GPU_MATHEMATICS::correlation<<<256,256>>>(s, order, ranks, filter, f_ld, signal, s_ld);
	}
		 //dimensional --movement correlation
	 template<typename number_type> static void cross_correlation_noAdjust(number_type* s, unsigned cor_mv, unsigned order, const  unsigned* store_ld, const number_type* filter,const  unsigned * f_ld,const  unsigned* f_ranks,
		    																							const number_type* signal, const unsigned * s_ld, const unsigned* s_ranks) {
		GPU_MATHEMATICS::cross_correlation_noAdjust<<<256,256>>>(s, cor_mv, order, store_ld, filter, f_ld, f_ranks, signal, s_ld, s_ranks);
	}

	 template<typename number_type> static void cross_correlation(number_type* s, unsigned cor_mv,
			 	 	 	 	 	 	 	 	 const  unsigned* store_ld, const number_type* filter,const  unsigned * f_ld,const  unsigned* f_ranks, unsigned f_order,
			 	 	 	 	 	 	 	 	 	 	 	 	 	 	    const number_type* signal, const unsigned * s_ld, const unsigned* s_ranks, unsigned s_order) {
		GPU_MATHEMATICS::cross_correlation<<<256,256>>>(s, cor_mv, store_ld, filter, f_ld, f_ranks, f_order, signal, s_ld, s_ranks, s_order);
	}
	 template<typename number_type> static void cross_correlation_filter_error(unsigned cor_mv, number_type* s, const  unsigned* store_ld,  const unsigned* store_ranks, unsigned store_order,
			 	 	 	 	 	 	 	 								const number_type* filter,const  unsigned * f_ld, const  unsigned* f_ranks, unsigned f_order,
			 	 	 	 	 	 	 	 	 	 	 	 	 	 	    const number_type* signal, const unsigned * s_ld, const unsigned* s_ranks, unsigned s_order) {
		GPU_MATHEMATICS::cc_filter_error<<<256,256>>>(cor_mv, s, store_ld, store_ranks, store_order, filter, f_ld, f_ranks, f_order, signal, s_ld, s_ranks, s_order);
	}

	 template<typename number_type> static void cross_correlation_signal_error(unsigned cor_mv, number_type* s, const  unsigned* store_ld,  const unsigned* store_ranks, unsigned store_order,
			 	 	 	 	 	 	 	 								const number_type* filter,const  unsigned * f_ld, const  unsigned* f_ranks, unsigned f_order,
			 	 	 	 	 	 	 	 	 	 	 	 	 	 	    const number_type* signal, const unsigned * s_ld, const unsigned* s_ranks, unsigned s_order) {
		GPU_MATHEMATICS::cc_signal_error<<<256,256>>>(cor_mv, s, store_ld, store_ranks, store_order, filter, f_ld, f_ranks, f_order, signal, s_ld, s_ranks, s_order);
	}


	 //Memory Management [written for preparation for writing with CUDA]]
	 template<typename number_type> static void initialize	(number_type*& d, unsigned sz) {
		GPU_MATHEMATICS::initialize<<<256,256>>>(d, sz);
	}
	 template<typename number_type> static void unified_initialize	(number_type*& d, unsigned sz) {
		GPU_MATHEMATICS::unified_initialize<<<256,256>>>(d, sz);
	}
	 template<typename number_type> static void destruction	(number_type* d) { GPU_MATHEMATICS::destruction(d); }

	 template<typename number_type> static void copy     	(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1, const unsigned* m1_LD) {
		GPU_MATHEMATICS::copy<<<256,256>>>(s, s_ranks, order, s_LD, m1, m1_LD);
	}

	 template<typename number_type> static void copy     	(number_type* s, const number_type* m, unsigned sz) {
		GPU_MATHEMATICS::copy<<<256,256>>>(s, m ,sz);
	}
	 template<typename number_type> static void fill     	(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, number_type m1) {
		GPU_MATHEMATICS::fill<<<256,256>>>(s, s_ranks, order, s_LD, m1);
	}
	 template<typename number_type> static void fill     	(number_type* s, number_type value, unsigned sz) {
		GPU_MATHEMATICS::fill<<<256,256>>>(s, value, sz);
	}
	 template<typename number_type> static void randomize     	(number_type* s, number_type lower_bound, number_type upper_bound, unsigned sz) {
#ifndef GPU_Tensor
		GPU_MATHEMATICS::randomize<<<256,256>>>(s, lower_bound, upper_bound, sz);
#endif
	}
	 template<typename number_type> static void transpose	(number_type* s, unsigned s_ld, const number_type* m, unsigned r, unsigned c, unsigned m_ld) {
		GPU_MATHEMATICS::transpose(s, s_ld, m, r, c, m_ld);
	}

	 template<typename number_type> static void power		(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type* m2,  const unsigned* m2_LD) {
		GPU_MATHEMATICS::power(s, s_ranks, order, s_LD, m1, m1_LD, m2, m2_LD);
	}
	 template<typename number_type> static void multiply	(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type* m2,  const unsigned* m2_LD) {
		GPU_MATHEMATICS::multiply(s, s_ranks, order, s_LD, m1, m1_LD, m2, m2_LD);
	}
	 template<typename number_type> static void divide	(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type* m2,  const unsigned* m2_LD) {
		GPU_MATHEMATICS::divide(s, s_ranks, order, s_LD, m1, m1_LD, m2, m2_LD);
	}
	 template<typename number_type> static void add		(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type* m2,  const unsigned* m2_LD) {
		GPU_MATHEMATICS::add(s, s_ranks, order, s_LD, m1, m1_LD, m2, m2_LD);
	}
	 template<typename number_type> static void subtract	(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type* m2,  const unsigned* m2_LD) {
		GPU_MATHEMATICS::subtract(s, s_ranks, order, s_LD, m1, m1_LD, m2, m2_LD);
	}
		    //By Scalar
	 template<typename number_type> static void power		(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type scal) {
		GPU_MATHEMATICS::power(s, s_ranks, order, s_LD, m1, m1_LD, scal);
	}
	 template<typename number_type> static void multiply	(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type scal) {
		GPU_MATHEMATICS::multiply(s, s_ranks, order, s_LD, m1, m1_LD, scal);
	}
	 template<typename number_type> static void divide	(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type scal) {
		GPU_MATHEMATICS::divide(s, s_ranks, order, s_LD, m1, m1_LD, scal);
	}
	 template<typename number_type> static void add		(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type scal) {
		GPU_MATHEMATICS::add(s, s_ranks, order, s_LD, m1, m1_LD, scal);
	}
	 template<typename number_type> static void subtract	(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type scal) {
		GPU_MATHEMATICS::subtract(s, s_ranks, order, s_LD, m1, m1_LD, scal);
	}
     template<typename number_type> static void print		(const number_type* ary, const unsigned* dims, const unsigned* lead_dims, unsigned index) {
    	GPU_MATHEMATICS::print(ary, dims, lead_dims, index);
    }
};

#endif /* GPU_CUH_ */
