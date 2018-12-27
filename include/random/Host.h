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
	    static void randomize(T& tensor, value_type lower_bound, value_type upper_bound) {
	 __BC_omp_for__
	        for (int i = 0; i < tensor.size(); ++i) {
	            tensor[i] = ((value_type) (std::rand() / ((value_type) RAND_MAX + 1)) * (upper_bound - lower_bound)) + lower_bound;
	        }
	 __BC_omp_bar__
	    }

		template<class value_type>
	    struct rand_handle {

	    	value_type lower_bound;
	    	value_type upper_bound;

	    	rand_handle(value_type lb, value_type ub)
	    		: lower_bound(lb), upper_bound(ub) {}

	        value_type operator () () const {
	           return (value_type(std::rand() / ((value_type)RAND_MAX)) * (upper_bound - lower_bound)) + lower_bound;
	        }

	        value_type operator [](unsigned i) {
	            return (value_type(std::rand() / ((value_type)RAND_MAX)) * (upper_bound - lower_bound)) + lower_bound;
	        }
	    };

		template<class value_type>
		static auto make_rand_gen(value_type lower, value_type upper) {
			return rand_handle<value_type>(lower, upper);
		}
};


}
}



#endif /* HOST_H_ */
