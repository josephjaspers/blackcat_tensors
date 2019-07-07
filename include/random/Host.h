/*
 * Host.h
 *
 *  Created on: Dec 3, 2018
 *      Author: joseph
 */

#ifndef BC_RANDOM_HOST_H_
#define BC_RANDOM_HOST_H_

#include <random>

namespace BC {
namespace random {

template<>
struct Random<host_tag> {
	 template<typename T, typename value_type>
	    static void randomize_kernel(T& tensor, value_type lower_bound, value_type upper_bound) {
	 BC_omp_for__
	        for (int i = 0; i < tensor.size(); ++i) {
	            tensor[i] = ((value_type) (std::rand() / ((value_type) RAND_MAX + 1)) * (upper_bound - lower_bound)) + lower_bound;
	        }
	 BC_omp_bar__
	    }

	 template<class Stream, typename T, typename value_type>
	    static void randomize(Stream stream, T& tensor, value_type lower_bound, value_type upper_bound) {
		 stream.enqueue([&](){
			 randomize_kernel(tensor, lower_bound, upper_bound);
		 });
	    }
};


}
}



#endif /* HOST_H_ */
