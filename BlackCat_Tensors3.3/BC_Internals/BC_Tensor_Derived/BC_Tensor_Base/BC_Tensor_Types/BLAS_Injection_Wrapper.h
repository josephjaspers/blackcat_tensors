/*
 * Injection_Info.h
 *
 *  Created on: Jun 9, 2018
 *      Author: joseph
 */

#ifndef INJECTION_INFO_H_
#define INJECTION_INFO_H_

namespace BC {
namespace internal {

template<class tensor_core, int alpha_modifier_ = 1, int beta_modifier_= 0>
struct injection_wrapper {

	injection_wrapper(tensor_core& array_) : array(array_) {}

	tensor_core& array;

	operator const tensor_core& () const { return array; }
	operator  	   tensor_core& ()       { return array; }

	 const tensor_core& data() const { return array; }
		   tensor_core& data()  { return array; }

};

template<int alpha, int beta, class tensor_core>
auto wrap_injection(tensor_core& tensor) {
	return injection_wrapper<tensor_core, alpha, beta>(tensor);
}


}
}



#endif /* INJECTION_INFO_H_ */
