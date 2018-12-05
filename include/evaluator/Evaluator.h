/*
 * Evaluator.h
 *
 *  Created on: Nov 25, 2018
 *      Author: joseph
 */

#ifndef EVALUATOR_H_
#define EVALUATOR_H_

#include "Device.cu"
#include "Host.h"

namespace BC {

class host_tag;
class device_tag;

namespace evaluator {
#ifdef __CUDACC__
	template<class system_tag>
	using implementation =
			std::conditional_t<
				std::is_same<host_tag, system_tag>::value,
					Host,
					Device>;
#else
	template<class system_tag>
	using implementation = Host;
#endif

}
}




#endif /* EVALUATOR_H_ */
