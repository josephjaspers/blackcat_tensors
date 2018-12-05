
/*
 * blas.h
 *
 *  Created on: Dec 3, 2018
 *      Author: joseph
 */

#ifndef BC_BLAS_BLAS_H_
#define BC_BLAS_BLAS_H_

#include "Device.h"
#include "Host.h"

namespace BC {

class host_tag;
class device_tag;

namespace blas {

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



#endif /* BLAS_H_ */
