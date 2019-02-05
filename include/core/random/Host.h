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

struct Host {
	 template<typename T, typename value_type>
	    static void randomize_kernel(T& tensor, value_type lower_bound, value_type upper_bound) {
	 __BC_omp_for__
	        for (int i = 0; i < tensor.size(); ++i) {
	            tensor[i] = ((value_type) (std::rand() / ((value_type) RAND_MAX + 1)) * (upper_bound - lower_bound)) + lower_bound;
	        }
	 __BC_omp_bar__
	    }

	 template<class Context, typename T, typename value_type>
	    static void randomize(Context context, T& tensor, value_type lower_bound, value_type upper_bound) {
		 context.push_job([&](){
			 randomize_kernel(tensor, lower_bound, upper_bound);
		 });
	    }
};


}
}



#endif /* HOST_H_ */
