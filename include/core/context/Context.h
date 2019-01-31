/*
 * Context.h
 *
 *  Created on: Jan 24, 2019
 *      Author: joseph
 */

#ifndef BC_CONTEXT_CONTEXT_H_
#define BC_CONTEXT_CONTEXT_H_

#include "Host.h"
#include "Device.h"


namespace BC {


class host_tag;
class device_tag;

namespace context {

#ifdef __CUDACC__
	template<class Allocator>
	using implementation =
			std::conditional_t<
				std::is_same<host_tag, typename BC::allocator_traits<Allocator>::system_tag>::value,
					Host<Allocator>,
					Device<Allocator>>;

#else
	template<
		class allocator,
		class=std::enable_if<std::is_same<host_tag, BC::allocator_traits<allocator>::system_tag>::value>
	>
	using implementation = Host<allocator>;
#endif

}
}



#endif /* CONTEXT_H_ */
