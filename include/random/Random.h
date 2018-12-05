/*
 * Random.h
 *
 *  Created on: Dec 3, 2018
 *      Author: joseph
 */

#ifndef RANDOM_H_
#define RANDOM_H_

#include "Host.h"
#include "Device.h"

namespace BC {

class host_tag;
class device_tag;

namespace random {

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



#endif /* RANDOM_H_ */
