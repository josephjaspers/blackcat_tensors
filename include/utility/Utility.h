/*
 * Utility.h
 *
 *  Created on: Dec 12, 2018
 *      Author: joseph
 */

#ifndef BC_UTILITY_UTILITY_H_
#define BC_UTILITY_UTILITY_H_


#include "Host.h"
#include "Device.h"

namespace BC {

class host_tag;
class device_tag;

namespace utility {


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




#endif /* UTILITY_H_ */
