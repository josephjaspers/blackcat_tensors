#include "BLACKCAT_CPU_MATHEMATICS.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"


template<typename number_type>
void  CPU_MATHEMATICS<number_type>::correlation(number_type* s, const number_type* filter, const number_type* signal, unsigned sz) {
    	number_type tmp = 0;
    	for (unsigned i = 0; i < sz; ++i) {
    		tmp += filter[i] * signal[i];
    	}
    	*s = tmp;
    }

template<typename number_type>
void CPU_MATHEMATICS<number_type>::correlation(number_type* s, unsigned order, const unsigned* ranks,const number_type* filter, const unsigned* f_ld,
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
void CPU_MATHEMATICS<number_type>::cross_correlation(number_type* s, unsigned cor_mv, const  unsigned* store_ld,
												const number_type* filter,const  unsigned * f_ld,const  unsigned* f_ranks, unsigned f_order,
												const number_type* signal, const unsigned * s_ld, const unsigned* s_ranks, unsigned s_order) {
	if (cor_mv == 0) {
		correlation(s, f_order, f_ranks, filter, f_ld, signal, s_ld);
	}

	else if (s_order > f_order) {
		--s_order;

		unsigned s_id = 0;
		for (unsigned i = 0; i < s_ranks[s_order]; ++i) {
					cross_correlation(s, cor_mv, store_ld, filter, f_ld, f_ranks, f_order, &signal[s_id], s_ld, s_ranks, s_order);

					s_id += s_ld[s_order];
				}
	}


	else {
		--cor_mv;

		unsigned store_id = 0;
		unsigned s_id = 0;

		unsigned mv_positions = s_ranks[cor_mv] + 1;
		mv_positions -= f_order < cor_mv + 1 ? 1 : f_ranks[cor_mv];

		for (unsigned i = 0; i < mv_positions; ++i) {
			cross_correlation(&s[store_id], cor_mv, store_ld, filter, f_ld, f_ranks, f_order, &signal[s_id], s_ld, s_ranks, s_order);

			store_id += store_ld[cor_mv];
			s_id += s_ld[cor_mv];
		}
	}
}


template<typename number_type>
void CPU_MATHEMATICS<number_type>::axpy(number_type* store, const unsigned* store_ld, const number_type* signal, const unsigned* signal_ld, const unsigned* signal_ranks, unsigned signal_order, number_type scalar) {

	--signal_order;

	if (signal_order == 0) {
		for (unsigned i = 0; i < signal_ranks[0]; ++i) {
			store[i] += signal[i] * scalar;
		//	std::cout << "multiplying signal[" << i << "] * " << scalar << " = " << signal[i] << " * " << scalar << " = " << signal[i] * scalar << " total = " << store[i] << std::endl;

		}
	} else {


		unsigned store_id = 0;
		unsigned sig_id = 0;
		//std::cout << " at order = " << signal_order + 1 <<std::endl;
//std::cout << " inital call ------------------ numbe pos "  << signal_ranks[signal_order] << std::endl;
		for (unsigned i = 0; i < signal_ranks[signal_order]; ++i) {
			//std::cout << " store_id = " << store_id << std::endl;
			//	std::cout << " sig_id = " << sig_id << std::endl;
			axpy(&store[store_id], store_ld, &signal[sig_id], signal_ld, signal_ranks, signal_order, scalar);
			store_id += store_ld[signal_order];
			sig_id += signal_ld[signal_order];

		}
	}
}


template<typename number_type>
void CPU_MATHEMATICS<number_type>::cc_filter_error(unsigned move_dimensions, number_type* store, const unsigned* store_ld, const unsigned* store_ranks, unsigned store_order,
										 	  const number_type* error, const unsigned* error_ld, const unsigned* error_ranks, unsigned error_order,
										 	  const number_type* signal,const unsigned* signal_ld, const unsigned* signal_ranks, unsigned signal_order) {

	if (move_dimensions > 0) {
		--move_dimensions;

		unsigned move_positions = signal_ranks[move_dimensions] - store_ranks[move_dimensions] + 1;
		unsigned sig_id = 0;
		unsigned err_id = 0;

		unsigned signal_base_index = 1;
		unsigned error_base_index = 1;

		for (int i = move_dimensions; i > 0; --i) {
			error_base_index *= error_ld[i];
			signal_base_index *= signal_ld[i];
		}


		for (unsigned i = 0; i < move_positions; ++i)  {
			cc_filter_error(move_dimensions, store, store_ld, store_ranks, store_order, &error[err_id], error_ld, error_ranks, error_order, &signal[sig_id], signal_ld, signal_ranks, signal_order);
			sig_id += signal_base_index;
			err_id += error_base_index;
		}
	} else {
		axpy(store, store_ld, signal, signal_ld, store_ranks, signal_order, *error);
	}
}


template<typename number_type>
void CPU_MATHEMATICS<number_type>::neg_axpy(number_type* store, const unsigned* store_ld, const number_type* signal, const unsigned* signal_ld, const unsigned* signal_ranks, unsigned signal_order, number_type scalar) {

	--signal_order;
std::cout << " neg_axpy " << std::endl;
	if (signal_order == 0) {
		for (unsigned i = 0; i < signal_ranks[0]; ++i) {
			store[i] -= signal[i] * scalar;
		//	std::cout << "multiplying signal[" << i << "] * " << scalar << " = " << signal[i] << " * " << scalar << " = " << signal[i] * scalar << " total = " << store[i] << std::endl;

		}
	} else {


		unsigned store_id = 0;
		unsigned sig_id = 0;
		//std::cout << " at order = " << signal_order + 1 <<std::endl;
//std::cout << " inital call ------------------ numbe pos "  << signal_ranks[signal_order] << std::endl;
		for (unsigned i = 0; i < signal_ranks[signal_order]; ++i) {
			//std::cout << " store_id = " << store_id << std::endl;
			//	std::cout << " sig_id = " << sig_id << std::endl;
			neg_axpy(&store[store_id], store_ld, &signal[sig_id], signal_ld, signal_ranks, signal_order, scalar);
			store_id += store_ld[signal_order];
			sig_id += signal_ld[signal_order];

		}
	}
}


template<typename number_type>
void CPU_MATHEMATICS<number_type>::cc_signal_error(unsigned move_dimensions, number_type* store, const unsigned* store_ld, const unsigned* store_ranks, unsigned store_order,
										 	  const number_type* error, const unsigned* error_ld, const unsigned* error_ranks, unsigned error_order,
										 	  const number_type* filter,const unsigned* filter_ld, const unsigned* filter_ranks, unsigned filter_order) {

	if (move_dimensions > 0) {
		--move_dimensions;

		unsigned move_positions = move_dimensions < error_order ?  error_ranks[move_dimensions] : 1;
		unsigned err_id = 0;
		unsigned sto_id = 0;

//		unsigned store_base_index = 1;
//		unsigned error_base_index = 1;
//
//		for (int i = move_dimensions; i > 0; --i) {
//			error_base_index *= error_ld[i];
//			store_base_index *= store_ld[i];
//		}

		for (unsigned i = 0; i < move_positions; ++i)  {
			cc_signal_error(move_dimensions, &store[sto_id], store_ld, store_ranks, store_order, &error[err_id], error_ld, error_ranks, error_order, filter, filter_ld, filter_ranks, filter_order);
			err_id += error_ld[move_dimensions];
			sto_id += store_ld[move_dimensions];
		}
	} else {
		axpy(store, store_ld, filter, filter_ld, filter_ranks, filter_order, *error);
	}
}
//
//template<typename number_type>
//void CPU_MATHEMATICS<number_type>::cc_error(unsigned move_dimensions, number_type* store, const unsigned* store_ld, const unsigned* store_ranks, unsigned store_order,
//										 	  const number_type* error, const unsigned* error_ld, const unsigned* error_ranks, unsigned error_order,
//										 	  const number_type* signal,const unsigned* signal_ld, const unsigned* signal_ranks, unsigned signal_order) {
//
//	if (move_dimensions == 0) {
////		std::cout << "calling axpy sub_kernel" << std::endl;
//		axpy(store, store_ld, signal, signal_ld, error_ranks, signal_order, error[0]);
//
//
//	} else if (signal_order > store_order) {
//		--signal_order;
//	//	std::cout << " moving loop ---------------------------------------" << std::endl;
//		unsigned sig_id = 0;
//		for (unsigned i = 0; i < signal_ranks[signal_order]; ++i) {
//			cc_error(move_dimensions, store, store_ld, store_ranks, store_order,
//						error, error_ld, error_ranks, error_order, &signal[sig_id], signal_ld, signal_ranks, signal_order);
//			sig_id += signal_ld[signal_order];
//		}
//	} else {
//		--move_dimensions;
//
//		unsigned mv_positions = signal_ranks[move_dimensions] + 1 - (store_order < move_dimensions ? 1 : store_ranks[move_dimensions]);
//		//mv_positions = 1;
////		std::cout << " the number of mv positions = " << mv_positions << std::endl;
////		std::cout << " move dim = " << move_dimensions << std::endl;
//		unsigned sig_id = 0;
//		unsigned store_id = 0;
//		unsigned error_id = 0;
//
//		for (unsigned i = 0; i < mv_positions; ++i) {
//			cc_error(move_dimensions, &store[store_id], store_ld, store_ranks, store_order,
//						&error[error_id], error_ld, error_ranks, error_order, &signal[sig_id], signal_ld, signal_ranks, signal_order);
//			sig_id += signal_ld[move_dimensions];
//			store_id += store_ld[move_dimensions];
//			error_id += error_ld[move_dimensions];
//		}
//	}
//
//}
