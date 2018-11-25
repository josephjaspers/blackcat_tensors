/*
 * algorithms.h
 *
 *  Created on: Nov 25, 2018
 *      Author: joseph
 */

#ifndef BC_AGORITHMS_ALGORITHMS_H_
#define BC_AGORITHMS_ALGORITHMS_H_

#include <type_traits>
#include "Device.h"
#include "Host.h"

namespace BC {

class host_tag;
class device_tag;

namespace algorithms {

#ifdef __CUDACC__
	template<class system_tag>
	using implementation =
			std::conditional_t<
				std::is_same<host_tag, system_tag>::value,
					host,
					device>;
#else
	template<class system_tag>
	using implementation = host;
#endif


}
}



#endif /* ALGORITHMS_H_ */
