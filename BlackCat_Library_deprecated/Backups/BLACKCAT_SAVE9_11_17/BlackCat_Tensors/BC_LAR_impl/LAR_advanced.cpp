#include "LinearAlgebraRoutines.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename number_type>
void  Tensor_Operations<number_type>::correlation(number_type* s, const number_type* filter, const number_type* signal, unsigned sz) {
    	number_type tmp = 0;
    	for (unsigned i = 0; i < sz; ++i) {
    		tmp += filter[i] * signal[i];
    	}
    	*s = tmp;
    }

template<typename number_type>
void Tensor_Operations<number_type>::correlation(number_type* s, unsigned order, const unsigned* ranks,const number_type* filter, const unsigned* f_ld,
																		      const number_type* signal, const unsigned* s_ld) {
	//WORKS
	--order;
	if (order == 0) {
    	number_type tmp = 0;
		correlation(&tmp, filter, signal, ranks[0]);
		*s += tmp;
	} else {

		for (unsigned i = 0; i < ranks[order]; ++i) {
			correlation(s, order, ranks, &filter[i * f_ld[order]], f_ld, &signal[i * s_ld[order]], s_ld);
		}
	}
}

template<typename number_type>
void Tensor_Operations<number_type>::cross_correlation(number_type* s, unsigned cor_mv, unsigned order, const  unsigned* store_ld, const number_type* filter,const  unsigned * f_ld,const  unsigned* f_ranks,
																										  const number_type* signal, const unsigned * s_ld, const unsigned* s_ranks) {
	if (cor_mv == 0) {
		correlation(s, order, f_ranks, filter, f_ld, signal, s_ld);
	} else {
		--cor_mv;

		unsigned store_id = 0;
		unsigned s_id = 0;
		for (unsigned i = 0; i < s_ranks[cor_mv] - f_ranks[cor_mv] + 1; ++i) {
			cross_correlation(&s[store_id], cor_mv, order, store_ld, filter, f_ld, f_ranks, &signal[s_id], s_ld, s_ranks);

			store_id += store_ld[cor_mv];
			s_id += s_ld[cor_mv];
		}
	}
}
