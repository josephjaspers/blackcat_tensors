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

#include "BLACKCAT_GPU_MATHEMATICS.cuh"

//template<typename number_type>
//__global__ void GPU_MATHEMATICS::initialize(number_type*& d, unsigned sz) {
//    d = new number_type[sz];
//}
//
//template<typename number_type>
//__global__ void GPU_MATHEMATICS::destruction(number_type * d) {
//    delete [] d;
//}

template<typename number_type>
__global__ void GPU_MATHEMATICS::copy(number_type * store, const number_type * v, unsigned sz) {
    for (int i = 0; i < sz; ++i) {
        store[i] = v[i];
    }
}

template<typename number_type>
__global__ void GPU_MATHEMATICS::fill(number_type * m, number_type f, unsigned sz) {
    for (int i = 0; i < sz; ++i) {
        m[i] = f;
    }
}

template<typename number_type>
__global__ void GPU_MATHEMATICS::transpose(number_type * s, unsigned s_ld, const number_type * m, unsigned rows, unsigned cols, unsigned m_ld) {

	for (unsigned r = 0; r < rows; ++r) {
		for (unsigned c= 0; c< cols; ++c) {
			s[r * s_ld + c] = m[c * m_ld + r];
		}
	}
}

template<typename number_type>
__global__ void GPU_MATHEMATICS::max(number_type* max_val, const number_type* data, const unsigned* ranks, const unsigned* ld, unsigned order) {

	if (order == 1) {
		for (unsigned i = 0; i < ranks[order-1]; ++i) {
			if (*max_val < data[i]) {
				*max_val = data[i];
			}
		}
	} else {

		for (unsigned i = 0; i < ranks[order-1]; ++i) {
			max(max_val, &data[i * ld[order-1]], ranks, ld, order-1);
		}
	}
}

template<typename number_type>
__global__ void GPU_MATHEMATICS::min(number_type* min_val, const number_type* data, const unsigned* ranks, const unsigned* ld, unsigned order) {

	if (order == 1) {
		for (unsigned i = 0; i < ranks[order-1]; ++i) {
			if (*min_val < data[i]) {
				*min_val = data[i];
			}
		}
	} else {

		for (unsigned i = 0; i < ranks[order-1]; ++i) {
			min(min_val, &data[i * ld[order-1]], ranks, ld, order-1);
		}
	}
}

template<typename number_type>
__global__ void GPU_MATHEMATICS::max_index(number_type* max_val, unsigned* max_indexes, const number_type* data, const unsigned* ranks, const unsigned* ld, unsigned order) {

	if (order == 1) {
		for (unsigned i = 0; i < ranks[order-1]; ++i) {
			if (*max_val < data[i]) {
				*max_val = data[i];
				*max_indexes = i;
			}
		}
	} else {

		number_type curr_max;
		for (unsigned i = 0; i < ranks[order-1]; ++i) {
			curr_max = *max_val;
			max_index(max_val, max_indexes, &data[i * ld[order-1]], ranks, ld, order-1);

			//if modified -- update
			if (curr_max != *max_val) {
				max_indexes[order-1] = i;
			}
		}
	}
}


template<typename number_type>
__global__ void GPU_MATHEMATICS::min_index(number_type* min_val, unsigned* min_indexes, const number_type* data, const unsigned* ranks, const unsigned* ld, unsigned order) {

	if (order == 1) {
		for (unsigned i = 0; i < ranks[order-1]; ++i) {
			if (*min_val < data[i]) {
				*min_val = data[i];
				*min_indexes = i;
			}
		}
	} else {

		number_type curr_max;
		for (unsigned i = 0; i < ranks[order-1]; ++i) {
			curr_max = *min_val;
			min_index(min_val, min_indexes, &data[i * ld[order-1]], ranks, ld, order-1);

			//if modified -- update
			if (curr_max != *min_val) {
				min_indexes[order-1] = i;
			}
		}
	}
}



template<typename number_type>
__global__ void GPU_MATHEMATICS::randomize(number_type * m, number_type lower_bound, number_type upper_bound, unsigned sz) {
    for (int i = 0; i < sz; ++i) {
        m[i] = rand() % (unsigned)(upper_bound - lower_bound) + lower_bound;
    }
}

template<typename number_type>
__global__ void GPU_MATHEMATICS::randomize(number_type * m, unsigned* ld, unsigned* ranks, unsigned order, number_type lower_bound, number_type upper_bound) {
    --order;
    if (order == 0) {
    	randomize(m, lower_bound, upper_bound , ranks[order]);
    } else {
    	for (unsigned i = 0; i < ranks[order]; ++i) {
    		randomize(&m[ld[order] * i], ld, ranks, order, lower_bound, upper_bound);
    	}
    }
}
template<typename number_type>
__global__ void GPU_MATHEMATICS::print(const number_type* ary, unsigned sz) {
	for (unsigned i = 0; i < sz; ++i) {
		std::cout << ary[i] << " " << std::endl;
	}
}

template<typename number_type>
__global__ void GPU_MATHEMATICS::print(const number_type* ary, const unsigned* dims, const unsigned* lead_dims, unsigned index) {


//	if (index < 3) {
//		for (unsigned r = 0; r < dims[0]; ++r) {
//
//			if (r != 0)
//			std::cout << std::endl;
//
//			for (unsigned c = 0; c< dims[1]; ++c) {
//				auto str =std::to_string(ary[r + c * lead_dims[index - 1]]);
//				str = str.substr(0, str.length() < 3 ? str.length() : 3);
//				std::cout << str << " ";
//			}
//		}
//		std::cout << "]" << std::endl << std::endl;
//
//	} else {
//		std::cout << "[";
//		for (unsigned i = 0; i < dims[index - 1]; ++i) {
//			print(&ary[i * lead_dims[index - 1]], dims, lead_dims, index - 1);
//		}
//	}
}
